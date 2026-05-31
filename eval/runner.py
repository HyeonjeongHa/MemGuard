#!/usr/bin/env python3
"""
Evaluation runner for MemGuard's two-call memory extraction pipeline.

Supports three datasets: LOCOMO, LongMemEval, and PerLTQA.

Usage examples
--------------
# Full build + answer with graph expansion (LOCOMO)
python -m eval.runner \\
  --dataset locomo \\
  --mode full \\
  --use_graph_expansion

# Answer-only (memories already built), with confidence-aware retrieval
python -m eval.runner \\
  --dataset locomo \\
  --mode answer \\
  --use_conf_aware_retrieval \\
  --use_graph_expansion
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from memguard_memory_system import MemorySystemFinal
from llm_client import OpenAIClient, create_llm_client
from config import (
    ANSWER_MODEL, JUDGE_MODEL, EMBEDDING_MODEL,
    QA_ANSWERING_TOP_K, OPENAI_API_KEY, OPENAI_BASE_URL,
)
from prompts import QUERY_MEMORY_ROUTE_PROMPT_RELATIONS, QUERY_WEIGHT_PROMPT_RELATIONS
from eval.llm_judge import LLMJudge
from eval.metrics import print_accuracy_stats
from eval.datasets import load_dataset, get_current_time


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_TYPE_PREFIXES = {
    "episodic":   "[EPISODIC]",
    "semantic":   "[SEMANTIC]",
    "procedural": "[PROCEDURAL]",
}
ROUTABLE_MEMORY_TYPES = ("semantic", "episodic", "procedural")


# ---------------------------------------------------------------------------
# Retrieval trace helpers
# ---------------------------------------------------------------------------

def get_memory_text(mem: Any) -> str:
    if isinstance(mem, dict):
        return str(mem.get("memory", mem.get("text", "")))
    return str(mem)


def infer_memory_type(memory_text: str) -> str:
    if not memory_text:
        return "unknown"
    text = memory_text.strip().upper()
    if text.startswith(MEMORY_TYPE_PREFIXES["episodic"].upper()):
        return "episodic"
    if text.startswith(MEMORY_TYPE_PREFIXES["semantic"].upper()):
        return "semantic"
    if text.startswith(MEMORY_TYPE_PREFIXES["procedural"].upper()):
        return "procedural"
    return "unknown"


def build_retrieval_trace(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for rank, mem in enumerate(memories, 1):
        if isinstance(mem, dict):
            score    = mem.get("score")
            metadata = mem.get("metadata", {})
        else:
            score    = None
            metadata = {}
        memory_text  = get_memory_text(mem)
        via_link     = mem.get("via_link")     if isinstance(mem, dict) else None
        expanded_from= mem.get("expanded_from")if isinstance(mem, dict) else None

        entry: Dict[str, Any] = {
            "rank":        rank,
            "memory_type": metadata['memory_type'],
            "memory":      memory_text,
            "score":       float(score) if isinstance(score, (int, float)) else None,
            "metadata":    metadata if isinstance(metadata, dict) else {},
        }
        if via_link:
            entry["via_link"] = via_link
        if expanded_from:
            entry["expanded_from"] = expanded_from
        trace.append(entry)
    return trace


def build_retrieval_stats(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"episodic": 0, "semantic": 0, "procedural": 0, "unknown": 0, "expanded": 0}
    for item in trace:
        mem_type = item.get("memory_type", "unknown")
        if mem_type not in stats:
            mem_type = "unknown"
        stats[mem_type] += 1
        if item.get("via_link"):
            stats["expanded"] += 1
    return stats


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

def parse_routed_memory_types(raw_output: str) -> List[str]:
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
                candidates = [value]; break
            if isinstance(value, list):
                candidates = [str(item) for item in value]; break

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
) -> tuple:
    if not isinstance(question, str) or not question.strip():
        return [], "", "empty_question"
    try:
        from jinja2 import Template
        template = Template(QUERY_MEMORY_ROUTE_PROMPT_RELATIONS)
        prompt   = template.render(user_query=question.strip())
        raw_output = memory.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        routed_types = parse_routed_memory_types(raw_output)
        return routed_types, raw_output, None
    except Exception as exc:
        return [], "", str(exc)


# ---------------------------------------------------------------------------
# Confidence-aware query routing
# ---------------------------------------------------------------------------

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
        from jinja2 import Template
        prompt = Template(QUERY_WEIGHT_PROMPT_RELATIONS).render(user_query=question.strip())
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

    except Exception as exc:
        print(f"  [conf-router] weight routing failed: {str(exc)[:80]}; using uniform weights")
        return default_weights


def get_retrieved_memories_conf_aware(
    memory: MemorySystemFinal,
    question: str,
    user_id: str,
    top_k: int,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    _stats_out: Optional[Dict] = None,
) -> tuple:
    t0 = time.time()
    weights = route_memory_type_weights(memory, question)
    if _stats_out is not None:
        _stats_out["weights"] = dict(weights)
    active = [t for t in ROUTABLE_MEMORY_TYPES if weights.get(t, 0.0) > 0]
    flat = _relation_aware_core(memory, question, top_k, weights, max_hops, hop_decay)
    source = "conf_aware[" + ",".join(f"{t}:{weights[t]:.2f}" for t in active) + "]"
    return flat, time.time() - t0, source


# ---------------------------------------------------------------------------
# Core retrieval with graph expansion
# ---------------------------------------------------------------------------

def get_retrieved_memories_with_expansion(
    memory: MemorySystemFinal,
    question: str,
    user_id: str,
    top_k: int,
    primary_types: Optional[List[str]] = None,
    max_hops: int = 1,
    hop_decay: float = 0.85,
) -> tuple:
    t0 = time.time()
    active = primary_types if primary_types else list(ROUTABLE_MEMORY_TYPES)
    weights = {t: 1.0 / len(active) for t in active}
    flat = _relation_aware_core(memory, question, top_k, weights, max_hops, hop_decay)
    return flat, time.time() - t0, "graph_expansion"


# ---------------------------------------------------------------------------
# Relation-aware retrieval — shared core + public wrappers
# ---------------------------------------------------------------------------

def _allocate_budget(weights: Dict[str, float], active_types: List[str], top_k: int) -> Dict[str, int]:
    """Distribute top_k slots across types proportionally using Hamilton largest-remainder."""
    raw       = {t: weights[t] * top_k for t in active_types}
    floored   = {t: int(raw[t]) for t in active_types}
    remainder = {t: raw[t] - floored[t] for t in active_types}
    deficit   = top_k - sum(floored.values())
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
    Weighted primary retrieval + per-primary graph expansion + budget-aware selection.

    Algorithm
    ---------
    1. Allocate top_k primary slots across active types proportionally to *weights*
       (Hamilton largest-remainder method).  Weights control only HOW MANY primaries
       come from each type — they do NOT alter the retrieval scores.
    2. Retrieve the allocated primaries from each type store (raw cosine scores).
    3. For each primary (sorted by raw score), BFS-expand via graph links up to
       *max_hops*.  Expansion order = link discovery order (not type-sorted).
       A memory claimed by an earlier group is skipped globally.
    4. Sort composed groups by raw primary score, then greedily include groups
       until the cumulative atomic count first reaches or exceeds top_k.
       The crossing group is always included in full.

    Returns a flat list ready for ``compose_memories_with_relations``:
    each selected group's primary followed by its BFS-ordered expansions.
    """
    import numpy as np
    import faiss as _faiss

    active_types = [t for t in ROUTABLE_MEMORY_TYPES if weights.get(t, 0.0) > 0]
    if not active_types:
        active_types = list(ROUTABLE_MEMORY_TYPES)
        weights = {t: 1.0 / len(ROUTABLE_MEMORY_TYPES) for t in ROUTABLE_MEMORY_TYPES}

    # Step 1 — budget allocation (weights → counts, not scores)
    n_per_type = _allocate_budget(weights, active_types, top_k)

    # Step 2 — retrieve primaries with raw cosine scores
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

    # Step 3 — query vector for expansion scoring
    query_emb = memory.embedding_func([question])[0]
    query_vec = np.array([query_emb], dtype=np.float32)
    _faiss.normalize_L2(query_vec)

    # Step 4 — BFS expansion per primary (discovery order, no type sort)
    globally_claimed: set = set(seen_ids)

    def bfs_expand(primary_id: str) -> List[Dict]:
        local_seen: set = set()
        frontier = [(primary_id, 1)]
        expansions: List[Dict] = []
        for _ in range(max_hops):
            next_frontier: List[tuple] = []
            for src_id, depth in frontier:
                for link in memory.cross_ref_graph.get_links(src_id):
                    rid   = link["related_id"]
                    rtype = link["memory_type"]
                    rel   = link["relation"]
                    if rid in globally_claimed or rid in local_seen:
                        continue
                    exp = memory._fetch_memory_by_id(rid, rtype)
                    if exp is None:
                        continue
                    cos = memory._score_memory_against_query(rid, rtype, query_vec)
                    exp = dict(exp)
                    exp["score"]         = cos * (hop_decay ** (depth - 1))
                    exp["via_link"]      = rel
                    exp["expanded_from"] = src_id
                    exp["hop_depth"]     = depth
                    expansions.append(exp)
                    local_seen.add(rid)
                    next_frontier.append((rid, depth + 1))
            frontier = next_frontier
        return expansions

    groups: List[Dict] = []
    for primary_mem in primaries:
        pid  = primary_mem.get("metadata", {}).get("memory_id")
        exps = bfs_expand(pid) if pid else []
        for exp in exps:
            eid = exp.get("metadata", {}).get("memory_id")
            if eid:
                globally_claimed.add(eid)
        groups.append({"primary": primary_mem, "expansions": exps})

    # Step 5 — score each group by embedding similarity of the full composed text.
    def _group_text(g: Dict) -> str:
        lines = [g["primary"].get("memory", "")]
        for exp in g["expansions"]:
            via = exp.get("via_link") or ""
            arrow = f"← {via[len('inverse_'):]}" if via.startswith("inverse_") else f"→ {via}"
            lines.append(f"  [{arrow}] {exp.get('memory', '')}")
        return "\n".join(lines)

    composed_texts = [_group_text(g) for g in groups]
    composed_embs  = memory.embedding_func(composed_texts)
    composed_vecs  = np.array(composed_embs, dtype=np.float32)
    _faiss.normalize_L2(composed_vecs)
    group_scores   = (composed_vecs @ query_vec[0]).tolist()

    for g, score in zip(groups, group_scores):
        g["score"] = float(score)

    groups.sort(key=lambda g: g["score"], reverse=True)
    selected = groups[:top_k]

    flat: List[Dict] = []
    for g in selected:
        flat.append(g["primary"])
        flat.extend(g["expansions"])
    return flat


def _relation_arrow(via_link: str) -> str:
    """Convert a via_link label to directional arrow notation."""
    if via_link.startswith("inverse_"):
        return f"← {via_link[len('inverse_'):]}"
    return f"→ {via_link}"


def compose_memories_with_relations(memories: List[Dict]) -> List[Dict]:
    """
    Create one composed entry per (primary, expansion) pair.

    Each primary memory is repeated once per direct expansion so that every
    composed entry carries exactly one relation link:

        [SEMANTIC] Alice enjoys hiking.
          [← elaborates] [EPISODIC] Alice went hiking with Bob last weekend.

    Primaries with no expansions appear as standalone entries.
    Orphan expansions (nodes whose parent is absent) appear prefixed with
    their relation arrow.
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

    consumed_ids: set = set()
    composed: List[Dict] = []

    for mem in memories:
        if mem.get("via_link"):
            continue

        mem_id            = mem.get("metadata", {}).get("memory_id")
        direct_expansions = children.get(mem_id, []) if mem_id else []

        if not direct_expansions:
            composed_mem = dict(mem)
            composed_mem["constituent_memories"] = [_atomic_entry(mem, "primary")]
            composed.append(composed_mem)
        else:
            for exp in direct_expansions:
                via   = exp.get("via_link") or ""
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
            via   = mem.get("via_link") or ""
            arrow = _relation_arrow(via)
            orphan = dict(mem)
            orphan["memory"] = f"[{arrow}] {mem.get('memory', '')}"
            orphan["constituent_memories"] = [_atomic_entry(mem, "orphan_expansion")]
            composed.append(orphan)
            consumed_ids.add(mid)

    return composed


# ---------------------------------------------------------------------------
# Memory building
# ---------------------------------------------------------------------------

def build_memories(
    memory: MemorySystemFinal,
    sample: Dict,
    user_id: str,
    max_sessions: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    sessions   = sample["haystack_sessions"]
    timestamps = sample.get("haystack_session_datetimes", [])

    if max_sessions:
        sessions   = sessions[:max_sessions]
        timestamps = timestamps[:max_sessions]

    print(f"\n📦 Building memories ({len(sessions)} sessions)")
    print("=" * 60)

    build_start    = time.time()
    total_memories = 0
    session_results= []

    for i, session in enumerate(sessions):
        if verbose:
            print(f"Processing session {i+1}/{len(sessions)}...", end=" ", flush=True)

        timestamp = timestamps[i] if i < len(timestamps) else None
        metadata  = {"timestamp": timestamp} if timestamp else None

        try:
            session_start = time.time()
            messages = (
                session.get("messages", session) if isinstance(session, dict) else session
            )
            result = memory.add(messages, user_id=user_id, metadata=metadata)
            session_time = time.time() - session_start

            if result and "results" in result:
                count = len(result["results"])
                total_memories += count
                session_results.append(
                    {
                        "session_id":    i,
                        "memories_count":count,
                        "timestamp":     timestamp,
                        "session_time":  session_time,
                    }
                )
                if verbose:
                    print(f"✅ {count} memories ({session_time:.1f}s)")
            else:
                if verbose:
                    print("⚠️ No results")
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            session_results.append({"session_id": i, "error": str(e)})

    build_time  = time.time() - build_start
    graph_stats = memory.cross_ref_graph.stats()

    print(f"\n✅ Memory build completed:")
    print(f"   Total memories:        {total_memories}")
    print(f"   Sessions processed:    {len(sessions)}")
    print(f"   Cross-reference links: {graph_stats['edges']} edges, {graph_stats['nodes']} nodes")
    print(f"   Total time:            {build_time:.1f} seconds")

    return {
        "total_memories":    total_memories,
        "sessions_processed":len(sessions),
        "build_time":        build_time,
        "session_results":   session_results,
        "user_id":           user_id,
        "graph_stats":       graph_stats,
    }


# ---------------------------------------------------------------------------
# Single-question answering
# ---------------------------------------------------------------------------

def _answer_single_question(
    memory: MemorySystemFinal,
    q: Dict,
    user_id: str,
    judge: LLMJudge,
    index: int,
    total: int,
    top_k: int = 10,
    current_time: str = None,
    use_routed_memory: bool = False,
    use_graph_expansion: bool = False,
    use_conf_aware_retrieval: bool = False,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    compose: bool = False,
    verbose: bool = False,
) -> Dict:
    result = {
        "question_id":    q["question_id"],
        "question":       q["question"],
        "expected_answer":q["answer"],
        "question_type":  q.get("question_type", "unknown"),
        "index":          index,
    }

    try:
        routed_types: List[str] = []
        routed_raw_output = ""
        routing_error: Optional[str] = None

        if use_routed_memory:
            routed_types, routed_raw_output, routing_error = route_required_memory_types(
                memory=memory,
                question=q["question"],
                temperature=0.0,
                max_tokens=128,
            )
            if verbose:
                if routing_error:
                    print(f"  Routing failed: {routing_error}")
                else:
                    print(f"  Routed types: {routed_types}")

        conf_aware_stats: Dict = {}
        primary_for_search = routed_types if (use_routed_memory and routed_types) else None

        if use_conf_aware_retrieval:
            memories_all, search_time, retrieval_source = get_retrieved_memories_conf_aware(
                memory=memory,
                question=q["question"],
                user_id=user_id,
                top_k=top_k,
                max_hops=max_hops,
                hop_decay=hop_decay,
                _stats_out=conf_aware_stats,
            )
            if verbose:
                w = conf_aware_stats.get("weights", {})
                expanded_count = sum(1 for m in memories_all if m.get("via_link"))
                print(f"  Conf weights: {w}, "
                      f"{expanded_count}/{len(memories_all)} via graph links")
        elif use_graph_expansion:
            memories_all, search_time, retrieval_source = get_retrieved_memories_with_expansion(
                memory=memory,
                question=q["question"],
                user_id=user_id,
                top_k=top_k,
                primary_types=primary_for_search,
                max_hops=max_hops,
                hop_decay=hop_decay,
            )
        else:
            search_start = time.time()
            if primary_for_search:
                results_by_type = memory.search_by_types(
                    q["question"], primary_for_search, top_k
                )
                memories_all = []
                for mem_list in results_by_type.values():
                    memories_all.extend(mem_list)
                memories_all.sort(
                    key=lambda x: float(x.get("score") or 0.0), reverse=True
                )
                retrieval_source = "search_routed"
            else:
                search_result = memory.search(q["question"], user_id=user_id, limit=top_k)
                memories_all  = (
                    search_result.get("results", [])
                    if isinstance(search_result, dict) else []
                )
                retrieval_source = "search_all"
            search_time = time.time() - search_start

        retrieval_trace_all = build_retrieval_trace(memories_all)
        retrieval_stats_all = build_retrieval_stats(retrieval_trace_all)

        if verbose:
            print(
                f"  Retrieved {len(memories_all)} memories ({search_time:.2f}s) "
                f"[{retrieval_source}]"
            )
            expanded_count = retrieval_stats_all.get("expanded", 0)
            if expanded_count:
                print(f"    ↳ {expanded_count} via graph expansion")
            for item in retrieval_trace_all:
                preview  = item["memory"][:120].replace("\n", " ")
                link_tag = f" [via {item.get('via_link', '')}]" if item.get("via_link") else ""
                print(f"    [{item['rank']}] {item['memory_type']}{link_tag} "
                      f"score={item['score']}: {preview}")

        use_compose = compose and (use_conf_aware_retrieval or use_graph_expansion)
        memories_for_answer = (
            compose_memories_with_relations(memories_all) if use_compose else memories_all
        )

        if use_compose and memories_for_answer:
            import numpy as np
            import faiss as _faiss
            texts = [m.get("memory", "") for m in memories_for_answer]
            composed_embs = memory.embedding_func(texts)
            query_emb     = memory.embedding_func([q["question"]])[0]
            composed_vecs = np.array(composed_embs, dtype=np.float32)
            query_vec     = np.array([query_emb],   dtype=np.float32)
            _faiss.normalize_L2(composed_vecs)
            _faiss.normalize_L2(query_vec)
            scores = (composed_vecs @ query_vec[0]).tolist()
            for m, s in zip(memories_for_answer, scores):
                m["composed_score"] = float(s)
            memories_for_answer = sorted(
                memories_for_answer,
                key=lambda m: m["composed_score"],
                reverse=True,
            )

        memories_for_answer = memories_for_answer[:top_k]

        if use_compose:
            id_to_retrieved_rank = {
                item["metadata"].get("memory_id"): item["rank"]
                for item in retrieval_trace_all
                if isinstance(item.get("metadata"), dict) and item["metadata"].get("memory_id")
            }
            context_trace = []
            for i, m in enumerate(memories_for_answer, 1):
                constituents = []
                for c in m.get("constituent_memories", []):
                    c = dict(c)
                    mid = c.get("memory_id")
                    if mid:
                        c["retrieved_rank"] = id_to_retrieved_rank.get(mid)
                    constituents.append(c)
                context_trace.append({
                    "rank":                 i,
                    "memory":               m.get("memory", ""),
                    "constituent_memories": constituents,
                })
        else:
            context_trace = None

        gen_start = time.time()
        answer    = memory.generate_answer(
            question=q["question"],
            memories=memories_for_answer,
            user_id=user_id,
            current_time=current_time,
        )
        gen_time = time.time() - gen_start

        if verbose:
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"  Generated: {answer_preview}")

        gt_answer   = q["answer"] or "Not Answerable"
        judge_start = time.time()
        llm_score   = judge.evaluate(q["question"], gt_answer, answer)
        judge_time  = time.time() - judge_start

        is_correct = llm_score == 1
        status     = "✅" if is_correct else "❌"

        result.update(
            {
                "generated_answer":            answer,
                "is_correct":                  is_correct,
                "llm_score":                   llm_score,
                "use_routed_memory":        bool(use_routed_memory),
                "use_graph_expansion":      bool(use_graph_expansion),
                "use_conf_aware_retrieval": bool(use_conf_aware_retrieval),
                "compose":                  bool(use_compose),
                "conf_aware_weights":       conf_aware_stats.get("weights"),
                "routed_memory_types":     routed_types,
                "routed_memory_raw_output":routed_raw_output,
                "routing_error":           routing_error,
                "retrieval_source":        retrieval_source,
                "memories_found":          len(memories_all),
                "context_entries":         len(memories_for_answer),
                "retrieved_memories":      retrieval_trace_all,
                "context_memories":        context_trace,
                "retrieval_stats":         retrieval_stats_all,
                "search_time":             search_time,
                "gen_time":                gen_time,
                "judge_time":              judge_time,
            }
        )

        if verbose:
            print(
                f"[{index}/{total}] {status} {q['question_id']}: "
                f"search={search_time:.1f}s gen={gen_time:.1f}s"
            )
        else:
            print(
                f"[{index}/{total}] {status} {q['question_id']}: "
                f"{search_time + gen_time:.1f}s"
            )

    except Exception as e:
        result["error"]      = str(e)
        result["llm_score"]  = 0
        result["is_correct"] = False
        print(f"[{index}/{total}] ❌ ERROR {q['question_id']}: {str(e)[:60]}")

    return result


# ---------------------------------------------------------------------------
# Batch question answering
# ---------------------------------------------------------------------------

def answer_questions(
    memory: MemorySystemFinal,
    questions: List[Dict],
    user_id: str,
    judge: LLMJudge,
    limit: int = None,
    top_k: int = 10,
    current_time: str = None,
    verbose: bool = True,
    concurrency: int = 1,
    use_routed_memory: bool = False,
    use_graph_expansion: bool = False,
    use_conf_aware_retrieval: bool = False,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    compose: bool = False,
) -> List[Dict]:
    if limit:
        questions = questions[:limit]
    total = len(questions)

    print(f"\n📝 Answering {total} questions "
          f"(routed={use_routed_memory}, graph_expansion={use_graph_expansion}, "
          f"conf_aware={use_conf_aware_retrieval}, compose={compose}, "
          f"max_hops={max_hops}, hop_decay={hop_decay})")
    print("=" * 60)

    def _run_one(args):
        q, idx = args
        return _answer_single_question(
            memory, q, user_id, judge,
            index=idx, total=total,
            top_k=top_k, current_time=current_time,
            use_routed_memory=use_routed_memory,
            use_graph_expansion=use_graph_expansion,
            use_conf_aware_retrieval=use_conf_aware_retrieval,
            max_hops=max_hops,
            hop_decay=hop_decay,
            compose=compose,
            verbose=verbose,
        )

    total_start = time.time()

    if concurrency > 1 and total > 1:
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_run_one, (q, i)): i
                for i, q in enumerate(questions, 1)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    idx = futures[future]
                    print(f"[{idx}/{total}] ❌ Concurrent error: {str(e)[:60]}")
        results.sort(key=lambda x: x.get("index", 0))
    else:
        results = [_run_one((q, i)) for i, q in enumerate(questions, 1)]

    total_time = time.time() - total_start
    print_accuracy_stats(results, "📊 Answer Summary")
    print(f"   Total time: {total_time:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Database path helper
# ---------------------------------------------------------------------------

def get_db_path(
    dataset_name: str,
    model_name: str,
    vector_store: str = "faiss",
    conversation_id: str = None,
    sample_id: str = None,
    character_id: str = None,
    self_check: bool = False,
) -> str:
    clean_model = model_name.replace("/", "_").replace(".", "_")
    sc_tag      = "_selfcheck" if self_check else ""
    base_dir    = f"{vector_store}_data_finalv2{sc_tag}"
    if conversation_id:
        return f"./{base_dir}/{dataset_name}/{clean_model}/{conversation_id}"
    elif sample_id:
        return f"./{base_dir}/{dataset_name}/{clean_model}/{sample_id}"
    elif character_id:
        return f"./{base_dir}/{dataset_name}/{clean_model}/{character_id}"
    return f"./{base_dir}/{dataset_name}/{clean_model}/default"


# ---------------------------------------------------------------------------
# Parallel worker — LongMemEval
# ---------------------------------------------------------------------------

def process_longmemeval_sample(sample_args):
    """Process a single LongMemEval sample in a subprocess (parallel mode)."""
    (sample, idx, total,
     model, model_name, judge_model,
     api_key, base_url, judge_api_key, judge_base_url,
     use_routed_memory, use_graph_expansion, use_conf_aware_retrieval,
     max_hops, hop_decay, compose, self_check_extraction,
     top_k, verbose, db_path, vector_store, mode, sessions) = sample_args

    sample_id = sample["question_id"]
    user_id   = f"test_{sample_id}"

    if sessions:
        sample = sample.copy()
        sample["haystack_sessions"] = sample["haystack_sessions"][:sessions]
        if "haystack_session_datetimes" in sample:
            sample["haystack_session_datetimes"] = sample["haystack_session_datetimes"][:sessions]

    sample_db_path = (
        os.path.join(db_path, sample_id) if db_path
        else get_db_path("longmemeval", model_name, vector_store,
                         sample_id=sample_id, self_check=self_check_extraction)
    )

    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    sample_llm_client = create_llm_client(model=model, **kwargs)

    sample_judge = LLMJudge(OpenAIClient(
        api_key=judge_api_key,
        base_url=judge_base_url,
        model=judge_model,
    ))

    memory = MemorySystemFinal(
        llm_client=sample_llm_client,
        self_check=self_check_extraction,
    )

    stub: Dict[str, Any] = {
        "question_id":    sample_id,
        "question":       sample["question"],
        "expected_answer": sample["answer"],
        "question_type":  sample.get("question_type", "unknown"),
        "index":          idx,
        "llm_score":      0,
        "is_correct":     False,
    }

    try:
        if mode in ("full", "build"):
            if os.path.exists(sample_db_path):
                memory.load(sample_db_path)
            else:
                build_memories(memory, sample, user_id, verbose=False)
                memory.save(sample_db_path)

        if mode == "answer":
            if not memory.load(sample_db_path):
                stub["error"] = f"No memory DB at {sample_db_path}"
                print(f"[{idx}/{total}] ⚠️  SKIP {sample_id}: memory DB not found")
                return stub

        if mode in ("full", "answer"):
            current_time = get_current_time(sample)
            question = {
                "question_id":   sample_id,
                "question":      sample["question"],
                "answer":        sample["answer"],
                "question_type": sample.get("question_type", "unknown"),
            }
            return _answer_single_question(
                memory=memory,
                q=question,
                user_id=user_id,
                judge=sample_judge,
                index=idx,
                total=total,
                top_k=top_k,
                current_time=current_time,
                use_routed_memory=use_routed_memory,
                use_graph_expansion=use_graph_expansion,
                use_conf_aware_retrieval=use_conf_aware_retrieval,
                max_hops=max_hops,
                hop_decay=hop_decay,
                compose=compose,
                verbose=verbose,
            )

    except Exception as e:
        stub["error"] = str(e)
        print(f"[{idx}/{total}] ❌ ERROR {sample_id}: {str(e)[:60]}")

    return stub


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args) -> int:
    model      = args.model or ANSWER_MODEL
    model_name = model if model.startswith("gpt") else args.model_name
    judge_model= args.judge_model or JUDGE_MODEL

    api_key  = args.api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    if not api_key:
        print("Error: OPENAI_API_KEY not configured.")
        return 1
    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    print("=" * 60)
    print("🧠 MemGuard — Two-Call Pipeline Evaluation")
    print(f"   Dataset:              {args.dataset}")
    print(f"   Mode:                 {args.mode}")
    print(f"   Model:                {model}")
    print(f"   Judge:                {judge_model}")
    print(f"   Routed memory:        {args.use_routed_memory}")
    print(f"   Graph expansion:      {args.use_graph_expansion}")
    print(f"   Conf-aware retrieval: {args.use_conf_aware_retrieval}")
    print(f"   Compose entries:      {args.compose}")
    print(f"   Max hops:             {args.max_hops}")
    print(f"   Hop decay:            {args.hop_decay}")
    print(f"   Self-check extract:   {args.self_check_extraction}")
    print("=" * 60)

    split_sample_ids = None
    if args.split and args.dataset == "longmemeval":
        splits_file = (
            Path(__file__).parent.parent / "data" / "longmemeval"
            / "splits" / "longmemeval_splits.json"
        )
        if not splits_file.exists():
            print(f"Error: Splits file not found: {splits_file}")
            return 1
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        if args.split not in splits_data:
            print(f"Error: Split '{args.split}' not found in {splits_file}")
            return 1
        split_sample_ids = splits_data[args.split]
        print(f"   Using '{args.split}' split: {len(split_sample_ids)} samples")

    print("\n📚 Loading data...")
    data = load_dataset(
        dataset_name=args.dataset,
        conv_id=args.conv_id,
        sample_id=args.sample_id,
        character_id=getattr(args, "character_id", None),
        num_samples=args.questions,
        sample_ids=split_sample_ids,
    )
    print(f"   Loaded {len(data)} samples")

    kwargs: Dict[str, Any] = {}
    if args.base_url: kwargs["base_url"] = args.base_url
    if args.api_key:  kwargs["api_key"]  = args.api_key

    llm_client = create_llm_client(model=model, **kwargs)

    judge_api_key  = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    judge_base_url = os.environ.get("OPENAI_BASE_URL")
    judge_client = OpenAIClient(
        api_key=judge_api_key,
        base_url=judge_base_url,
        model=judge_model,
    )
    judge = LLMJudge(judge_client)
    main_embedding_func = None

    expansion_tag = (
        f"_hops-{args.max_hops}_decay-{args.hop_decay}"
        if args.use_graph_expansion else ""
    )
    k_tag = f"_k-{args.top_k}" if args.top_k != QA_ANSWERING_TOP_K else ""

    compose_tag = "_composed" if args.compose else ""
    if args.self_check_extraction:
        run_tag = (
            f"routed-{args.use_routed_memory}_graph-{args.use_graph_expansion}"
            f"_conf-{args.use_conf_aware_retrieval}{compose_tag}_selfcheck_{expansion_tag}{k_tag}"
        )
    else:
        run_tag = (
            f"routed-{args.use_routed_memory}_graph-{args.use_graph_expansion}"
            f"_conf-{args.use_conf_aware_retrieval}{compose_tag}{expansion_tag}{k_tag}"
        )

    # ------------------------------------------------------------------
    # LOCOMO
    # ------------------------------------------------------------------
    if args.dataset == "locomo":
        all_conv_ids = sorted(set(d["conversation_id"] for d in data))
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir   = Path(args.output_dir) / run_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        all_results  = []

        for i, conv_id in enumerate(all_conv_ids, 1):
            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(all_conv_ids)}] {conv_id}")
            print(f"{'='*60}")

            save_path = result_dir / f"{conv_id}_results.json"
            if args.mode == "answer" and save_path.exists():
                print("   Already answered — loading cached result")
                with open(save_path) as f:
                    all_results.append(json.load(f))
                continue

            conv_questions = [d for d in data if d["conversation_id"] == conv_id]
            conv_sample    = conv_questions[0]
            user_id        = f"test_{conv_id}"

            if args.sessions:
                conv_sample = conv_sample.copy()
                conv_sample["haystack_sessions"] = conv_sample["haystack_sessions"][:args.sessions]
                conv_sample["haystack_session_datetimes"] = conv_sample[
                    "haystack_session_datetimes"][:args.sessions]

            conv_db_path = (
                os.path.join(args.db_path, conv_id) if args.db_path
                else get_db_path("locomo", model_name, args.vector_store,
                                 conversation_id=conv_id, self_check=args.self_check_extraction)
            )
            print(f"   📁 {conv_db_path}")

            memory = MemorySystemFinal(
                llm_client=llm_client,
                embedding_func=main_embedding_func,
                self_check=args.self_check_extraction,
            )

            if args.mode in ("full", "build"):
                if os.path.exists(conv_db_path):
                    print("   Memory already exists — loading")
                    memory.load(conv_db_path)
                else:
                    build_result = build_memories(
                        memory, conv_sample, user_id,
                        max_sessions=args.sessions, verbose=args.verbose,
                    )
                    print(f"   ✅ {build_result['total_memories']} memories, "
                          f"{build_result['graph_stats']['edges']} graph edges")
                    memory.save(conv_db_path)

            elif args.mode == "answer":
                if not memory.load(conv_db_path):
                    print("   ⚠️  No memory DB — run --mode build first")
                    continue

            if args.mode in ("full", "answer"):
                current_time = get_current_time(conv_sample)
                if current_time:
                    print(f"   📅 Dataset timeline: {current_time}")

                results = answer_questions(
                    memory, conv_questions, user_id, judge,
                    limit=args.questions, top_k=args.top_k,
                    current_time=current_time, verbose=args.verbose,
                    concurrency=args.concurrency,
                    use_routed_memory=args.use_routed_memory,
                    use_graph_expansion=args.use_graph_expansion,
                    use_conf_aware_retrieval=args.use_conf_aware_retrieval,
                    max_hops=args.max_hops, hop_decay=args.hop_decay,
                    compose=args.compose,
                )

                valid = [r for r in results if "llm_score" in r]
                conv_result = {
                    "conversation_id": conv_id,
                    "questions":       len(valid),
                    "correct":         sum(r["llm_score"] for r in valid),
                    "accuracy":        round(
                        sum(r["llm_score"] for r in valid) / len(valid) * 100, 2
                    ) if valid else 0,
                    "results": results,
                }
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(conv_result, f, indent=2, ensure_ascii=False)
                all_results.append(conv_result)

        if all_results:
            total_q = sum(r["questions"] for r in all_results)
            total_c = sum(r["correct"]   for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            summary = {
                "timestamp":               timestamp,
                "dataset":                 args.dataset,
                "model":                   model,
                "judge_model":             judge_model,
                "use_routed_memory":       bool(args.use_routed_memory),
                "use_graph_expansion":     bool(args.use_graph_expansion),
                "use_conf_aware_retrieval":bool(args.use_conf_aware_retrieval),
                "total_conversations":     len(all_results),
                "total_questions":         total_q,
                "total_correct":           total_c,
                "overall_accuracy":        f"{acc:.2f}%",
                "by_conversation": {r["conversation_id"]: f"{r['accuracy']:.2f}%" for r in all_results},
            }
            with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n{'='*60}")
            print(f"📊 Overall: {acc:.2f}% ({total_c}/{total_q})")
            print(f"💾 Results saved to: {result_dir}/")

    # ------------------------------------------------------------------
    # LongMemEval
    # ------------------------------------------------------------------
    elif args.dataset == "longmemeval":
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / run_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        all_results: List[Dict] = []

        if args.parallel and len(data) > 1:
            print(f"\n🚀 LongMemEval parallel processing ({args.workers} workers, {len(data)} samples)")
            print("=" * 60)

            _api_key  = kwargs.get("api_key")  or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
            _base_url = kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL

            sample_args_list = [
                (
                    sample, i, len(data),
                    model, model_name, judge_model,
                    _api_key, _base_url, judge_api_key, judge_base_url,
                    args.use_routed_memory, args.use_graph_expansion, args.use_conf_aware_retrieval,
                    args.max_hops, args.hop_decay, args.compose, args.self_check_extraction,
                    args.top_k, args.verbose, args.db_path, args.vector_store, args.mode,
                    args.sessions,
                )
                for i, sample in enumerate(data, 1)
            ]

            from multiprocessing import Pool
            with Pool(processes=args.workers) as pool:
                raw_results = pool.map(process_longmemeval_sample, sample_args_list)

            raw_results.sort(key=lambda x: x.get("index", 0))
            for r in raw_results:
                sid = r.get("question_id")
                if sid:
                    with open(result_dir / f"{sid}_results.json", "w", encoding="utf-8") as f:
                        json.dump([r], f, indent=2, ensure_ascii=False)
            all_results.extend(raw_results)

        else:
            for i, sample in enumerate(data, 1):
                sample_id = sample["question_id"]
                user_id   = f"test_{sample_id}"

                print(f"\n{'='*60}")
                print(f"🔹 [{i}/{len(data)}] {sample_id}")
                print(f"{'='*60}")

                save_path = result_dir / f"{sample_id}_results.json"
                if args.mode in ["full", "answer"] and save_path.exists():
                    print("   Already answered — loading cached result")
                    with open(save_path) as f:
                        cached = json.load(f)
                    all_results.extend(cached if isinstance(cached, list) else [cached])
                    continue

                sample_db_path = (
                    os.path.join(args.db_path, sample_id) if args.db_path
                    else get_db_path("longmemeval", model_name, args.vector_store,
                                     sample_id=sample_id, self_check=args.self_check_extraction)
                )
                print(f"   📁 {sample_db_path}")

                if args.sessions:
                    sample = sample.copy()
                    sample["haystack_sessions"] = sample["haystack_sessions"][:args.sessions]
                    if "haystack_session_datetimes" in sample:
                        sample["haystack_session_datetimes"] = sample["haystack_session_datetimes"][:args.sessions]

                memory = MemorySystemFinal(
                    llm_client=llm_client,
                    embedding_func=main_embedding_func,
                    self_check=args.self_check_extraction,
                )

                if args.mode in ("full", "build"):
                    if os.path.exists(sample_db_path):
                        print("   Memory already exists — loading")
                        memory.load(sample_db_path)
                    else:
                        build_result = build_memories(
                            memory, sample, user_id,
                            max_sessions=args.sessions, verbose=args.verbose,
                        )
                        print(f"   ✅ {build_result['total_memories']} memories, "
                              f"{build_result['graph_stats']['edges']} graph edges")
                        memory.save(sample_db_path)

                elif args.mode == "answer":
                    if not memory.load(sample_db_path):
                        print("   ⚠️  No memory DB — run --mode build first")
                        continue

                if args.mode in ("full", "answer"):
                    current_time = get_current_time(sample)
                    question = {
                        "question_id":   sample["question_id"],
                        "question":      sample["question"],
                        "answer":        sample["answer"],
                        "question_type": sample.get("question_type", "unknown"),
                    }
                    results = answer_questions(
                        memory, [question], user_id, judge,
                        top_k=args.top_k, current_time=current_time,
                        verbose=args.verbose,
                        use_routed_memory=args.use_routed_memory,
                        use_graph_expansion=args.use_graph_expansion,
                        use_conf_aware_retrieval=args.use_conf_aware_retrieval,
                        max_hops=args.max_hops, hop_decay=args.hop_decay,
                        compose=args.compose,
                    )
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    all_results.extend(results)

        if all_results:
            total_q = len(all_results)
            total_c = sum(r.get("llm_score", 0) for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            summary = {
                "timestamp":               timestamp,
                "dataset":                 args.dataset,
                "model":                   model,
                "use_routed_memory":       bool(args.use_routed_memory),
                "use_graph_expansion":     bool(args.use_graph_expansion),
                "use_conf_aware_retrieval":bool(args.use_conf_aware_retrieval),
                "total_samples":           total_q,
                "total_correct":           total_c,
                "overall_accuracy":        f"{acc:.2f}%",
            }
            with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(result_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print_accuracy_stats(all_results, "📊 LongMemEval overall results")
            print(f"💾 Results saved to: {result_dir}/")

    # ------------------------------------------------------------------
    # PerLTQA
    # ------------------------------------------------------------------
    elif args.dataset == "perltqa":
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / run_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for i, character in enumerate(data, 1):
            char_id   = character["character_id"]
            char_name = character.get("character_name", char_id)
            user_id   = f"test_{char_id}"
            questions = character.get("questions", [])

            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(data)}] {char_name} ({char_id})")
            print(f"{'='*60}")

            save_path = result_dir / f"{char_id}_results.json"
            if args.mode == "answer" and save_path.exists():
                print("   Already answered — loading cached result")
                with open(save_path) as f:
                    all_results.append(json.load(f))
                continue

            char_db_path = (
                os.path.join(args.db_path, char_id) if args.db_path
                else get_db_path("perltqa", model_name, args.vector_store,
                                 character_id=char_id, self_check=args.self_check_extraction)
            )
            print(f"   📁 {char_db_path}")

            if args.sessions:
                character = character.copy()
                character["haystack_sessions"] = character["haystack_sessions"][:args.sessions]
                if "haystack_session_datetimes" in character:
                    character["haystack_session_datetimes"] = character[
                        "haystack_session_datetimes"][:args.sessions]

            memory = MemorySystemFinal(
                llm_client=llm_client,
                embedding_func=main_embedding_func,
                self_check=args.self_check_extraction,
            )

            if args.mode in ("full", "build"):
                if os.path.exists(char_db_path):
                    print("   Memory already exists — loading")
                    memory.load(char_db_path)
                else:
                    build_result = build_memories(
                        memory, character, user_id,
                        max_sessions=args.sessions, verbose=args.verbose,
                    )
                    print(f"   ✅ {build_result['total_memories']} memories, "
                          f"{build_result['graph_stats']['edges']} graph edges")
                    memory.save(char_db_path)

            elif args.mode == "answer":
                if not memory.load(char_db_path):
                    print("   ⚠️  No memory DB — run --mode build first")
                    continue

            if args.mode in ("full", "answer"):
                current_time = get_current_time(character)
                results = answer_questions(
                    memory, questions, user_id, judge,
                    limit=args.questions, top_k=args.top_k,
                    current_time=current_time, verbose=args.verbose,
                    concurrency=args.concurrency,
                    use_routed_memory=args.use_routed_memory,
                    use_graph_expansion=args.use_graph_expansion,
                    use_conf_aware_retrieval=args.use_conf_aware_retrieval,
                    max_hops=args.max_hops, hop_decay=args.hop_decay,
                    compose=args.compose,
                )
                valid = [r for r in results if "llm_score" in r]
                char_result = {
                    "character_id":   char_id,
                    "character_name": char_name,
                    "questions":      len(valid),
                    "correct":        sum(r["llm_score"] for r in valid),
                    "accuracy":       round(
                        sum(r["llm_score"] for r in valid) / len(valid) * 100, 2
                    ) if valid else 0,
                    "results": results,
                }
                with open(result_dir / f"{char_id}_results.json", "w", encoding="utf-8") as f:
                    json.dump(char_result, f, indent=2, ensure_ascii=False)
                all_results.append(char_result)

        if all_results:
            total_q = sum(r["questions"] for r in all_results)
            total_c = sum(r["correct"]   for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            summary = {
                "timestamp":               timestamp,
                "dataset":                 args.dataset,
                "model":                   model,
                "use_routed_memory":       bool(args.use_routed_memory),
                "use_graph_expansion":     bool(args.use_graph_expansion),
                "use_conf_aware_retrieval":bool(args.use_conf_aware_retrieval),
                "total_characters":        len(all_results),
                "total_questions":         total_q,
                "total_correct":           total_c,
                "overall_accuracy":        f"{acc:.2f}%",
                "by_character": {
                    r["character_id"]: f"{r['accuracy']:.2f}%" for r in all_results
                },
            }
            with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n{'='*60}")
            print(f"📊 PerLTQA: {acc:.2f}% ({total_c}/{total_q})")
            print(f"💾 Results saved to: {result_dir}/")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MemGuard — Two-Call Pipeline Evaluation Runner"
    )

    # Dataset
    parser.add_argument("--dataset", choices=["locomo", "longmemeval", "perltqa"],
                        default="locomo")
    parser.add_argument("--conv-id",       default=None)
    parser.add_argument("--sample-id",     default=None)
    parser.add_argument("--character-id",  default=None)
    parser.add_argument("--split", choices=["sft", "rl", "test"], default=None,
                        help="Use a predefined LongMemEval split from "
                             "data/longmemeval/splits/longmemeval_splits.json")

    # Mode
    parser.add_argument("--mode", choices=["build", "answer", "full"], default="full")

    # Limits
    parser.add_argument("--sessions",  type=int, default=None)
    parser.add_argument("--questions", type=int, default=None)

    # Model
    parser.add_argument("--model",       default=None)
    parser.add_argument("--model_name",  default=None)
    parser.add_argument("--judge-model", default=None)

    # API
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key",  default=None)

    # Retrieval
    parser.add_argument("--top-k", type=int, default=QA_ANSWERING_TOP_K)
    parser.add_argument("--use_routed_memory",       action="store_true")
    parser.add_argument("--use_graph_expansion",     action="store_true")
    parser.add_argument("--use_conf_aware_retrieval", action="store_true",
                        help="Confidence-aware routing: LLM assigns per-type budget weights")
    parser.add_argument("--compose", action="store_true",
                        help="Merge each primary memory and its graph-linked expansions into "
                             "a single composed entry for the answering LLM.")
    parser.add_argument("--max-hops",  type=int,   default=1,    dest="max_hops")
    parser.add_argument("--hop-decay", type=float, default=0.85, dest="hop_decay")
    parser.add_argument("--self-check-extraction",
                        dest="self_check_extraction",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Run an additional LLM pass after Call 1 to find missed facts "
                             "(on by default; use --no-self-check-extraction to disable)")

    # Persistence
    parser.add_argument("--db-path",      default=None)
    parser.add_argument("--vector-store", default="faiss", choices=["faiss_v2"])

    # Concurrency
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel sample processing (LongMemEval)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Number of parallel workers for LongMemEval sample processing")

    # Output
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--verbose",    action="store_true")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.model_name or "default")
    return run_evaluation(args)


if __name__ == "__main__":
    exit(main())
