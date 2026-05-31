"""
Evaluation runner using MemGuard's two-call memory extraction pipeline
for the HaluMem benchmark.

Usage
-----
# Default: conf-aware retrieval ON, graph expansion ON
python HaluMem/eval/eval_memguard.py \\
  --model gpt-4.1-mini --version v1

# Graph expansion only (no conf-aware routing)
python HaluMem/eval/eval_memguard.py \\
  --model gpt-4.1-mini --use_graph_expansion --version v1-graph
"""

import os
import re
import sys
import time
import json
import copy
import traceback
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

_HERE = Path(__file__).parent          # HaluMem/eval
_ROOT = Path(__file__).parent.parent.parent  # MemGuard-Release project root

sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu
from jinja2 import Template


def _import_local(name: str):
    spec = _ilu.spec_from_file_location(f"halumem_{name}", _HERE / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llms_mod = _import_local("llms")
_prompts_mod = _import_local("prompts")

llm_request = _llms_mod.llm_request
PROMPT_MEMBUILDER = _prompts_mod.PROMPT_MEMBUILDER

from tqdm import tqdm
from memguard_memory_system import MemorySystemFinal
from llm_client import create_llm_client, OpenAIClient
from config import OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL
from prompts import QUERY_MEMORY_ROUTE_PROMPT_RELATIONS, QUERY_WEIGHT_PROMPT_RELATIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTABLE_MEMORY_TYPES = ("semantic", "episodic", "procedural")

TEMPLATE_MEMROUTER = """Retrieved Memories (Semantic):
{semantic_memories}

Retrieved Memories (Episodic):
{episodic_memories}

Retrieved Memories (Procedural):
{procedural_memories}
"""

TEMPLATE_MEMROUTER_COMPOSED = """Memories for user:
{memories}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_memory_type(memory_text: str) -> str:
    if not memory_text:
        return "unknown"
    text = memory_text.strip().upper()
    if text.startswith("[EPISODIC]"):
        return "episodic"
    if text.startswith("[SEMANTIC]"):
        return "semantic"
    if text.startswith("[PROCEDURAL]"):
        return "procedural"
    return "unknown"


# ---------------------------------------------------------------------------
# Compose helpers
# ---------------------------------------------------------------------------

def _relation_arrow(via_link: str) -> str:
    if via_link.startswith("inverse_"):
        return f"← {via_link[len('inverse_'):]}"
    return f"→ {via_link}"


def _atomic_entry(mem: Dict, role: str) -> Dict:
    meta = mem.get("metadata", {}) if isinstance(mem.get("metadata"), dict) else {}
    return {
        "memory_id":   meta.get("memory_id"),
        "memory_type": meta.get("memory_type"),
        "memory":      mem.get("memory", ""),
        "score":       mem.get("score"),
        "role":        role,
        "via_link":    mem.get("via_link"),
        "hop_depth":   mem.get("hop_depth"),
    }


def compose_memories_with_relations(memories: List[Dict]) -> List[Dict]:
    """
    Create one composed entry per (primary, expansion) pair.

    Each primary memory is repeated once per direct expansion so that every
    composed entry carries exactly one relation link:

        [SEMANTIC] Alice enjoys hiking.
          [← elaborates] [EPISODIC] Alice went hiking with Bob last weekend.

    Primaries with no expansions appear as standalone entries.
    Orphan expansions appear prefixed with their relation arrow.
    """
    all_ids: set = {
        mem.get("metadata", {}).get("memory_id")
        for mem in memories
        if mem.get("metadata", {}).get("memory_id")
    }

    children: Dict[str, List[Dict]] = {}
    for mem in memories:
        parent_id = mem.get("expanded_from")
        if parent_id and parent_id in all_ids:
            children.setdefault(parent_id, []).append(mem)

    consumed_ids: set = set()
    composed: List[Dict] = []

    for mem in memories:
        if mem.get("via_link"):
            continue

        mem_id = mem.get("metadata", {}).get("memory_id")
        direct_expansions = children.get(mem_id, []) if mem_id else []

        if not direct_expansions:
            composed_mem = dict(mem)
            composed_mem["constituent_memories"] = [_atomic_entry(mem, "primary")]
            composed.append(composed_mem)
        else:
            for exp in direct_expansions:
                via = exp.get("via_link") or ""
                arrow = _relation_arrow(via)
                exp_id = exp.get("metadata", {}).get("memory_id")
                if exp_id:
                    consumed_ids.add(exp_id)
                composed_mem = dict(mem)
                composed_mem["memory"] = "\n".join([
                    mem.get("memory", ""),
                    f"  [{arrow}] {exp.get('memory', '')}",
                ])
                composed_mem["constituent_memories"] = [
                    _atomic_entry(mem, "primary"),
                    _atomic_entry(exp, "expansion"),
                ]
                composed.append(composed_mem)

        if mem_id:
            consumed_ids.add(mem_id)

    for mem in memories:
        mid = mem.get("metadata", {}).get("memory_id")
        if mid and mid not in consumed_ids:
            via = mem.get("via_link") or ""
            arrow = _relation_arrow(via)
            orphan = dict(mem)
            orphan["memory"] = f"[{arrow}] {mem.get('memory', '')}"
            orphan["constituent_memories"] = [_atomic_entry(mem, "orphan_expansion")]
            composed.append(orphan)
            consumed_ids.add(mid)

    return composed


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

def _parse_routed_memory_types(raw_output: str) -> List[str]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return []
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates: List[str] = []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, str):
        candidates = [parsed]
    elif isinstance(parsed, list):
        candidates = [str(item) for item in parsed]
    elif isinstance(parsed, dict):
        for key in ("routing", "memory_type", "memory_types", "type", "types"):
            value = parsed.get(key)
            if isinstance(value, str):
                candidates = [value]
                break
            if isinstance(value, list):
                candidates = [str(item) for item in value]
                break

    if not candidates:
        matches = re.findall(r"\b(semantic|episodic|procedural)\b", text.lower())
        if matches:
            candidates = matches

    seen: set = set()
    normalized: List[str] = []
    for c in candidates:
        token = str(c).strip().lower()
        if token in ROUTABLE_MEMORY_TYPES and token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def route_required_memory_types(
    memory: MemorySystemFinal,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> Tuple[List[str], str, Optional[str]]:
    if not isinstance(question, str) or not question.strip():
        return [], "", "empty_question"
    try:
        template = Template(QUERY_MEMORY_ROUTE_PROMPT_RELATIONS)
        prompt = template.render(user_query=question.strip())
        raw_output = memory.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        routed_types = _parse_routed_memory_types(raw_output)
        return routed_types, raw_output, None
    except Exception as exc:
        return [], "", str(exc)


def route_memory_type_weights(
    memory: MemorySystemFinal,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> Dict[str, float]:
    default_weights = {t: 1.0 / len(ROUTABLE_MEMORY_TYPES) for t in ROUTABLE_MEMORY_TYPES}
    if not question or not question.strip():
        return default_weights
    try:
        template = Template(QUERY_WEIGHT_PROMPT_RELATIONS)
        prompt = template.render(user_query=question.strip())
        raw = memory.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not raw:
            return default_weights
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(clean)
        weights_raw = parsed.get("weights", {})
        weights: Dict[str, float] = {}
        for t in ROUTABLE_MEMORY_TYPES:
            w = weights_raw.get(t, 0.0)
            weights[t] = float(w) if isinstance(w, (int, float)) else 0.0
        total = sum(weights.values())
        if total <= 0:
            return default_weights
        return {t: w / total for t, w in weights.items()}
    except Exception:
        return default_weights


# ---------------------------------------------------------------------------
# Relation-aware retrieval core
# ---------------------------------------------------------------------------

def _allocate_budget(weights: Dict[str, float], active_types: List[str], top_k: int) -> Dict[str, int]:
    """Distribute top_k slots proportionally to weights using Hamilton largest-remainder."""
    raw = {t: weights[t] * top_k for t in active_types}
    floored = {t: int(raw[t]) for t in active_types}
    remainder = {t: raw[t] - floored[t] for t in active_types}
    deficit = top_k - sum(floored.values())
    for t in sorted(active_types, key=lambda t: remainder[t], reverse=True)[:deficit]:
        floored[t] += 1
    return floored


def _relation_aware_core(
    memory: MemorySystemFinal,
    question: str,
    top_k: int,
    weights: Dict[str, float],
    max_hops: int,
    hop_decay: float,
) -> List[Dict]:
    """
    Weighted primary retrieval + BFS graph expansion + group-level cosine reranking.

    Algorithm
    ---------
    1. Allocate top_k primary slots across active types proportionally to weights
       (Hamilton largest-remainder method).
    2. Retrieve the allocated primaries from each type store (raw cosine scores).
    3. For each primary, BFS-expand via graph links up to max_hops.
    4. Score each group by embedding similarity of the full composed text, then
       sort by that score and select top_k groups.

    Returns a flat list: each selected group's primary followed by its expansions.
    """
    import numpy as np
    import faiss as _faiss

    active_types = [t for t in ROUTABLE_MEMORY_TYPES if weights.get(t, 0.0) > 0]
    if not active_types:
        active_types = list(ROUTABLE_MEMORY_TYPES)
        weights = {t: 1.0 / len(ROUTABLE_MEMORY_TYPES) for t in ROUTABLE_MEMORY_TYPES}

    n_per_type = _allocate_budget(weights, active_types, top_k)

    primaries: List[Dict] = []
    seen_ids: set = set()
    for mem_type in active_types:
        n = n_per_type.get(mem_type, 0)
        if n == 0:
            continue
        store = memory.vector_stores.get(mem_type)
        if store is None:
            continue
        for mem in memory._ensure_metadata_type(store.search(question, n), mem_type):
            mid = mem.get("metadata", {}).get("memory_id")
            if mid and mid not in seen_ids:
                primaries.append(dict(mem))
                seen_ids.add(mid)

    query_emb = memory.embedding_func([question])[0]
    query_vec = np.array([query_emb], dtype=np.float32)
    _faiss.normalize_L2(query_vec)

    globally_claimed: set = set(seen_ids)

    def bfs_expand(primary_id: str) -> List[Dict]:
        local_seen: set = set()
        frontier = [(primary_id, 1)]
        expansions: List[Dict] = []
        for _ in range(max_hops):
            next_frontier: List[tuple] = []
            for src_id, depth in frontier:
                for link in memory.cross_ref_graph.get_links(src_id):
                    rid = link["related_id"]
                    rtype = link["memory_type"]
                    rel = link["relation"]
                    if rid in globally_claimed or rid in local_seen:
                        continue
                    exp = memory._fetch_memory_by_id(rid, rtype)
                    if exp is None:
                        continue
                    cos = memory._score_memory_against_query(rid, rtype, query_vec)
                    exp = dict(exp)
                    exp["score"] = cos * (hop_decay ** (depth - 1))
                    exp["via_link"] = rel
                    exp["expanded_from"] = src_id
                    exp["hop_depth"] = depth
                    expansions.append(exp)
                    local_seen.add(rid)
                    next_frontier.append((rid, depth + 1))
            frontier = next_frontier
        return expansions

    groups: List[Dict] = []
    for primary_mem in primaries:
        pid = primary_mem.get("metadata", {}).get("memory_id")
        exps = bfs_expand(pid) if pid else []
        for exp in exps:
            eid = exp.get("metadata", {}).get("memory_id")
            if eid:
                globally_claimed.add(eid)
        groups.append({"primary": primary_mem, "expansions": exps})

    def _group_text(g: Dict) -> str:
        lines = [g["primary"].get("memory", "")]
        for exp in g["expansions"]:
            via = exp.get("via_link") or ""
            arrow = f"← {via[len('inverse_'):]}" if via.startswith("inverse_") else f"→ {via}"
            lines.append(f"  [{arrow}] {exp.get('memory', '')}")
        return "\n".join(lines)

    if groups:
        composed_texts = [_group_text(g) for g in groups]
        composed_embs = memory.embedding_func(composed_texts)
        composed_vecs = np.array(composed_embs, dtype=np.float32)
        _faiss.normalize_L2(composed_vecs)
        group_scores = (composed_vecs @ query_vec[0]).tolist()
        for g, score in zip(groups, group_scores):
            g["score"] = float(score)

    groups.sort(key=lambda g: g["score"], reverse=True)
    selected = groups[:top_k]

    flat: List[Dict] = []
    for g in selected:
        flat.append(g["primary"])
        flat.extend(g["expansions"])
    return flat


# ---------------------------------------------------------------------------
# Retrieval wrapper
# ---------------------------------------------------------------------------

def search_memories_final(
    memory: MemorySystemFinal,
    query: str,
    user_id: str,
    top_k: int = 10,
    use_routed_memory: bool = True,
    use_graph_expansion: bool = True,
    use_conf_aware_routing: bool = False,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    compose: bool = False,
    _stats_out: Optional[Dict] = None,
) -> Tuple[str, List[str], float, Dict]:
    """Retrieve memories and build a structured context string.

    Returns (context_str, memory_texts, duration_ms, routing_info).
    """
    start = time.time()

    routed_types: List[str] = []
    routed_raw: str = ""
    routing_error: Optional[str] = None
    conf_weights: Optional[Dict[str, float]] = None
    retrieval_source: str = "search_all"
    memories: List[Dict] = []

    if use_conf_aware_routing:
        conf_weights = route_memory_type_weights(memory=memory, question=query)
        active_types = [t for t in ROUTABLE_MEMORY_TYPES if conf_weights.get(t, 0.0) > 0]
        source_parts = [f"{t}:{conf_weights[t]:.2f}" for t in active_types]
        retrieval_source = "conf_aware[" + ",".join(source_parts) + "]"
        memories = _relation_aware_core(memory, query, top_k, conf_weights, max_hops, hop_decay)
    elif use_graph_expansion:
        if use_routed_memory:
            routed_types, routed_raw, routing_error = route_required_memory_types(memory, query)
        active = routed_types if (use_routed_memory and routed_types) else list(ROUTABLE_MEMORY_TYPES)
        weights = {t: 1.0 / len(active) for t in active}
        memories = _relation_aware_core(memory, query, top_k, weights, max_hops, hop_decay)
        retrieval_source = "graph_expansion"
    elif use_routed_memory:
        routed_types, routed_raw, routing_error = route_required_memory_types(memory, query)
        if routed_types:
            results_by_type = memory.search_by_types(query, routed_types, top_k)
            for mem_list in results_by_type.values():
                memories.extend(mem_list)
            memories.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            retrieval_source = "search_routed"
        else:
            search_result = memory.search(query, user_id=user_id, limit=top_k)
            memories = search_result.get("results", []) if isinstance(search_result, dict) else []
    else:
        search_result = memory.search(query, user_id=user_id, limit=top_k)
        memories = search_result.get("results", []) if isinstance(search_result, dict) else []

    duration_ms = (time.time() - start) * 1000

    use_compose = compose and (use_conf_aware_routing or use_graph_expansion)
    memories_for_context = compose_memories_with_relations(memories) if use_compose else memories

    if use_compose and memories_for_context:
        import numpy as np
        import faiss as _faiss
        texts = [m.get("memory", "") for m in memories_for_context]
        composed_embs = memory.embedding_func(texts)
        query_emb = memory.embedding_func([query])[0]
        composed_vecs = np.array(composed_embs, dtype=np.float32)
        query_vec = np.array([query_emb], dtype=np.float32)
        _faiss.normalize_L2(composed_vecs)
        _faiss.normalize_L2(query_vec)
        scores = (composed_vecs @ query_vec[0]).tolist()
        for m, s in zip(memories_for_context, scores):
            m["composed_score"] = float(s)
        memories_for_context = sorted(
            memories_for_context,
            key=lambda m: m["composed_score"],
            reverse=True,
        )

    memories_for_context = memories_for_context[:top_k]

    if use_compose:
        lines = [
            m.get("memory", str(m)) if isinstance(m, dict) else str(m)
            for m in memories_for_context
        ]
        context = TEMPLATE_MEMROUTER_COMPOSED.format(memories="\n".join(lines) or "[No memories found]")
    else:
        typed: Dict[str, List[str]] = {"semantic": [], "episodic": [], "procedural": []}
        for m in memories_for_context:
            text = m.get("memory", str(m)) if isinstance(m, dict) else str(m)
            mem_type = _infer_memory_type(text)
            typed.get(mem_type, typed["semantic"]).append(text)
        context = TEMPLATE_MEMROUTER.format(
            semantic_memories="\n".join(typed["semantic"]) or "[No memories found]",
            episodic_memories="\n".join(typed["episodic"]) or "[No memories found]",
            procedural_memories="\n".join(typed["procedural"]) or "[No memories found]",
        )

    memory_texts = [
        m.get("memory", str(m)) if isinstance(m, dict) else str(m)
        for m in memories_for_context
    ]

    routing_info = {
        "routed_types": routed_types,
        "routed_raw": routed_raw,
        "routing_error": routing_error,
        "retrieval_source": retrieval_source,
        "conf_weights": conf_weights,
    }
    if _stats_out is not None:
        _stats_out.update(routing_info)

    return context, memory_texts, duration_ms, routing_info


# ---------------------------------------------------------------------------
# Memory system factory
# ---------------------------------------------------------------------------

def _create_memory_system(model: str, self_check: bool = False) -> MemorySystemFinal:
    llm_client = create_llm_client(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=model,
    )
    return MemorySystemFinal(llm_client=llm_client, self_check=self_check)


# ---------------------------------------------------------------------------
# Dialogue ingestion
# ---------------------------------------------------------------------------

def add_dialogue(
    memory: MemorySystemFinal,
    dialogue: list,
    user_id: str,
    session_start_time: str,
) -> Tuple[List[str], float]:
    date_format = "%b %d, %Y, %H:%M:%S"
    dt = datetime.strptime(session_start_time, date_format).replace(tzinfo=timezone.utc)

    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in dialogue
    ]
    metadata = {"timestamp": dt.isoformat(), "unix_timestamp": int(dt.timestamp())}

    start = time.time()
    result = memory.add(messages, user_id=user_id, metadata=metadata)
    duration_ms = (time.time() - start) * 1000

    extracted_memories = [r["memory"] for r in result.get("results", [])]
    return extracted_memories, duration_ms


# ---------------------------------------------------------------------------
# Per-user evaluation
# ---------------------------------------------------------------------------

def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        return match.group(1).strip()
    raise ValueError("No name found in persona_info.")


def process_user(
    user_data: dict,
    top_k: int,
    save_path: str,
    model: str,
    use_routed_memory: bool = True,
    use_graph_expansion: bool = True,
    use_conf_aware_routing: bool = False,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    self_check: bool = True,
    compose: bool = False,
):
    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    memory = _create_memory_system(model=model, self_check=self_check)
    _qa_llm_request = llm_request

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    try:
        for session in tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}"):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"],
            }

            dialogue = session["dialogue"]
            start_time = session.get(
                "start_time",
                dialogue[0]["timestamp"] if dialogue else "Jan 01, 2020, 00:00:00",
            )

            extracted_memories, duration_ms = add_dialogue(
                memory=memory,
                dialogue=dialogue,
                user_id=user_name,
                session_start_time=start_time,
            )

            new_session["add_dialogue_duration_ms"] = duration_ms

            if session.get("is_generated_qa_session", False):
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            new_session["extracted_memories"] = extracted_memories

            for memory_point in new_session["memory_points"]:
                if memory_point["is_update"] == "False" or not memory_point.get("original_memories"):
                    continue
                stats_out: Dict = {}
                _, memories_from_system, _, _ = search_memories_final(
                    memory=memory,
                    query=memory_point["memory_content"],
                    user_id=user_name,
                    top_k=top_k,
                    use_routed_memory=use_routed_memory,
                    use_graph_expansion=use_graph_expansion,
                    use_conf_aware_routing=use_conf_aware_routing,
                    max_hops=max_hops,
                    hop_decay=hop_decay,
                    compose=compose,
                    _stats_out=stats_out,
                )
                memory_point["memories_from_system"] = memories_from_system

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []

            for qa in session["questions"]:
                stats_out: Dict = {}
                context, _, search_duration_ms, routing_info = search_memories_final(
                    memory=memory,
                    query=qa["question"],
                    user_id=user_name,
                    top_k=top_k,
                    use_routed_memory=use_routed_memory,
                    use_graph_expansion=use_graph_expansion,
                    use_conf_aware_routing=use_conf_aware_routing,
                    max_hops=max_hops,
                    hop_decay=hop_decay,
                    compose=compose,
                    _stats_out=stats_out,
                )

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = search_duration_ms
                new_qa["retrieval_source"] = routing_info["retrieval_source"]
                new_qa["routed_types"] = routing_info["routed_types"]
                if routing_info.get("conf_weights"):
                    new_qa["conf_weights"] = routing_info["conf_weights"]
                if routing_info.get("routing_error"):
                    new_qa["routing_error"] = routing_info["routing_error"]

                prompt = PROMPT_MEMBUILDER.format(
                    context=context,
                    question=qa["question"],
                )

                t0 = time.time()
                new_qa["system_response"] = _qa_llm_request(prompt)
                new_qa["response_duration_ms"] = (time.time() - t0) * 1000

                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False)

        print(f"Saved user {user_name} to {tmp_file}")
        return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}

    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{user_data['uuid']}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"Error in user {user_name}: {e}")
        return {"uuid": user_data["uuid"], "status": "error", "path": error_path}


def iter_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    max_workers: int = 1,
    model: str = "gpt-4.1",
    model_name: str = None,
    use_routed_memory: bool = True,
    use_graph_expansion: bool = True,
    use_conf_aware_routing: bool = False,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    self_check: bool = True,
    compose: bool = False,
):
    frame = "memguard"
    model_label = model_name or (model.split("/")[-1] if "/" in model else model)
    compose_tag = "_composed" if compose else ""
    routing_tag = (
        f"routed-{use_routed_memory}_graph-{use_graph_expansion}"
        f"_conf-{use_conf_aware_routing}{compose_tag}_topk{top_k}"
        f"_hops{max_hops}_decay{hop_decay}"
    )
    save_path = f"results/{frame}-{model_label}-{routing_tag}-{version}/"
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Model              : {model_label}")
    print(f"Use routed memory  : {use_routed_memory}")
    print(f"Use graph expansion: {use_graph_expansion}")
    print(f"Conf-aware routing : {use_conf_aware_routing}")
    print(f"Max hops           : {max_hops}")
    print(f"Hop decay          : {hop_decay}")
    print(f"Self-check extract : {self_check}")
    print(f"Compose memories   : {compose}")
    print(f"Output dir         : {save_path}")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        idx = 0
        for idx, user_data in enumerate(iter_jsonl(data_path), 1):
            uuid = user_data["uuid"]
            future = executor.submit(
                process_user,
                user_data,
                top_k,
                save_path,
                model,
                use_routed_memory,
                use_graph_expansion,
                use_conf_aware_routing,
                max_hops,
                hop_decay,
                self_check,
                compose,
            )
            futures[future] = uuid

        total_users = idx

        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] Finished {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] Error processing {uuid}: {e}")
                traceback.print_exc()

    with open(output_file, "a", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json"):
                file_path = os.path.join(tmp_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Skipped {file}: {e}")

    elapsed = time.time() - start_time
    print(f"All done in {elapsed:.2f}s")
    print(f"Final results saved to: {output_file}")


if __name__ == "__main__":
    import argparse  # noqa: F811  (already in stdlib, re-import is cheap)

    parser = argparse.ArgumentParser(
        description="HaluMem evaluation with MemGuard (two-call extraction pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str,
                        default=str(_HERE.parent / "data" / "HaluMem-Medium.jsonl"))
    parser.add_argument("--version", type=str, default="default",
                        help="Run label appended to the output directory name.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. gpt-4.1-mini).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short model name for the results directory.")

    # Retrieval options
    parser.add_argument("--use_routed_memory", action="store_true",
                        help="Route query to primary memory type(s) before retrieval.")
    parser.add_argument("--use_graph_expansion", action="store_true",
                        help="Expand primary retrieval via cross-reference graph.")
    parser.add_argument("--use_conf_aware_routing", action="store_true",
                        help="LLM assigns per-type confidence weights; supersedes --use_routed_memory.")
    parser.add_argument("--max_hops", type=int, default=1,
                        help="BFS hops to follow from primary results.")
    parser.add_argument("--hop_decay", type=float, default=0.85,
                        help="Score multiplier applied per hop beyond the first.")
    parser.add_argument("--self_check", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Run self-check extraction pass after Call 1 (on by default; "
                             "use --no-self_check to disable).")
    parser.add_argument("--compose", action="store_true",
                        help="Merge each primary memory with its direct graph expansion into "
                             "one context block per expansion.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        version=args.version,
        top_k=args.top_k,
        max_workers=args.max_workers,
        model=args.model,
        model_name=args.model_name,
        use_routed_memory=args.use_routed_memory,
        use_graph_expansion=args.use_graph_expansion,
        use_conf_aware_routing=args.use_conf_aware_routing,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        self_check=args.self_check,
        compose=args.compose,
    )
