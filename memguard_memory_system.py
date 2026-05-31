"""
MemGuard Memory System

Three-tier memory architecture with a cross-reference graph and two-call extraction pipeline.

Classes (bottom to top):
  VectorStore              – FAISS-backed per-type memory store
  MemorySystemBase         – Core storage, search, and answer generation
  MemorySystemWithGraph    – Extends base with CrossReferenceGraph and ID index
  MemorySystemFinal        – Two-call extraction pipeline (Call 1 + Call 2)

Usage:
    from memguard_memory_system import MemorySystemFinal
    memory = MemorySystemFinal(llm_client=client)
    memory.add(messages, user_id="user1", metadata={"timestamp": "2024-01-01"})
    results = memory.search("What do I like to eat?", user_id="user1")
    memory.save("./faiss_db/user1")
    memory.load("./faiss_db/user1")
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from jinja2 import Template

from config import MEMORY_CONSTRUCTION_TOP_K, QA_ANSWERING_TOP_K, DEFAULT_TOP_K
from prompts import (
    PROMPT_A_EXTRACT_RELATE_ROUTE,
    PROMPT_B_ASSIGN_OPS_LINKS,
    PROMPT_C_SELF_CHECK_EXTRACTION,
)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """FAISS-backed vector store for a single memory type."""

    def __init__(self, embedding_func, dimension: int = 1536):
        import faiss
        self.embedding_func = embedding_func
        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.memories: List[str] = []
        self.metadata: List[Dict] = []

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._faiss.write_index(self.index, os.path.join(output_dir, "index.faiss"))
        with open(os.path.join(output_dir, "payload.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"dimension": self.dimension, "memories": self.memories, "metadata": self.metadata},
                f, ensure_ascii=False, indent=2,
            )

    def load(self, input_dir: str) -> None:
        index_path   = os.path.join(input_dir, "index.faiss")
        payload_path = os.path.join(input_dir, "payload.json")
        if not os.path.exists(index_path) or not os.path.exists(payload_path):
            raise FileNotFoundError(f"VectorStore files not found in: {input_dir}")
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dimension = int(payload.get("dimension", self.dimension))
        self.index     = self._faiss.read_index(index_path)
        self.memories  = list(payload.get("memories", []))
        self.metadata  = list(payload.get("metadata", []))

    def add(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        import numpy as np
        vectors = np.array(self.embedding_func(texts), dtype=np.float32)
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.memories.extend(texts)
        self.metadata.extend(metadatas if metadatas else [{} for _ in texts])

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.memories:
            return []
        import numpy as np
        query_vec = np.array(self.embedding_func([query]), dtype=np.float32)
        self._faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, min(top_k, len(self.memories)))
        return [
            {"memory": self.memories[idx], "score": float(score), "metadata": self.metadata[idx]}
            for score, idx in zip(scores[0], indices[0]) if idx >= 0
        ]


# ---------------------------------------------------------------------------
# CrossReferenceGraph
# ---------------------------------------------------------------------------

class CrossReferenceGraph:
    """
    Bidirectional adjacency map: memory_id → list of link records.

    Each link record: {"related_id": str, "relation": str, "memory_type": str}

    Adding A→B also stores B→A with "inverse_" prefix on the relation.
    """

    def __init__(self):
        self._graph: Dict[str, List[Dict[str, str]]] = {}

    def add_link(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        source_type: str,
        target_type: str,
    ) -> None:
        self._graph.setdefault(source_id, []).append(
            {"related_id": target_id, "relation": relation, "memory_type": target_type}
        )
        self._graph.setdefault(target_id, []).append(
            {"related_id": source_id, "relation": f"inverse_{relation}", "memory_type": source_type}
        )

    def get_links(self, memory_id: str) -> List[Dict[str, str]]:
        return self._graph.get(memory_id, [])

    def has_node(self, memory_id: str) -> bool:
        return memory_id in self._graph

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._graph, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._graph = json.load(f)

    def node_count(self) -> int:
        return len(self._graph)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._graph.values()) // 2

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.node_count(), "edges": self.edge_count()}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_memory_id(mem_type: str, memory_text: str) -> str:
    """Return a stable, type-prefixed SHA-1 hash ID for a memory string."""
    digest = hashlib.sha1(memory_text.encode("utf-8")).hexdigest()[:12]
    return f"{mem_type[:3].lower()}_{digest}"


# ---------------------------------------------------------------------------
# MemorySystemBase
# ---------------------------------------------------------------------------

class MemorySystemBase:
    """
    Core memory system: FAISS stores for semantic/episodic/procedural memories,
    plus search and answer generation.
    """

    def __init__(self, llm_client, embedding_func=None):
        self.llm_client = llm_client
        self.embedding_func = embedding_func or (lambda texts: llm_client.get_embeddings(texts))
        self.prompts: Dict[str, str] = {}

        self.semantic_store   = VectorStore(self.embedding_func)
        self.episodic_store   = VectorStore(self.embedding_func)
        self.procedural_store = VectorStore(self.embedding_func)
        self.vector_stores: Dict[str, VectorStore] = {
            "semantic":   self.semantic_store,
            "episodic":   self.episodic_store,
            "procedural": self.procedural_store,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, db_path: str) -> None:
        os.makedirs(db_path, exist_ok=True)
        vector_store_dir = os.path.join(db_path, "vector_store")
        for mem_type, store in self.vector_stores.items():
            store.save(os.path.join(vector_store_dir, mem_type))
        print(f"Saved memories to: {db_path}")
        for mem_type, store in self.vector_stores.items():
            print(f"  Vector index ({mem_type}): {len(store.memories)} memories")

    def load(self, db_path: str) -> bool:
        vector_store_dir = os.path.join(db_path, "vector_store")
        if not os.path.isdir(vector_store_dir):
            print(f"Warning: memory database not found: {db_path}")
            return False
        typed_loaded = False
        for mem_type, store in self.vector_stores.items():
            typed_dir = os.path.join(vector_store_dir, mem_type)
            if (os.path.exists(os.path.join(typed_dir, "index.faiss")) and
                    os.path.exists(os.path.join(typed_dir, "payload.json"))):
                store.load(typed_dir)
                typed_loaded = True
        if not typed_loaded:
            print(f"Warning: no typed memory indices found in: {vector_store_dir}")
            return False
        print(f"Loaded memories from: {db_path}")
        for mem_type, store in self.vector_stores.items():
            print(f"  Vector index ({mem_type}): {len(store.memories)} memories")
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, user_id: str = None, limit: int = None, top_k: int = DEFAULT_TOP_K) -> Dict:
        k = limit if limit is not None else top_k
        combined: List[Dict] = []
        for mem_type, store in self.vector_stores.items():
            combined.extend(self._ensure_metadata_type(store.search(query, k), mem_type))
        combined.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return {"results": combined[:k]}

    def search_by_types(
        self,
        query: str,
        memory_types: List[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, List[Dict]]:
        return {
            mem_type: self._ensure_metadata_type(
                self.vector_stores[mem_type].search(query, top_k), mem_type
            )
            for mem_type in memory_types
            if mem_type in self.vector_stores
        }

    def _ensure_metadata_type(self, results: List[Dict], memory_type: str) -> List[Dict]:
        out: List[Dict] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata", {})
            if isinstance(meta, dict) and "memory_type" not in meta:
                item = dict(item)
                item["metadata"] = {**meta, "memory_type": memory_type}
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        memories: List[Dict] = None,
        user_id: str = None,
        current_time: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        if memories is None:
            memories = self.search(question, user_id, top_k=QA_ANSWERING_TOP_K).get("results", [])

        context = ""
        if additional_context:
            context += f"{additional_context}\n\n"
        context += "Retrieved Memories:"
        for i, mem in enumerate(memories, 1):
            memory_text = mem.get("memory", mem.get("text", str(mem))) if isinstance(mem, dict) else str(mem)
            context += f"{i}. {memory_text}\n"

        time_context = (
            f"\n\n**The current date/time is {current_time}. Use this as the reference point "
            f"when answering questions about relative time (e.g., 'last year', 'yesterday', 'recently').**"
        ) if current_time else ""

        prompt = f"""{context}{time_context}

Question: {question}

Instructions:
1. Carefully analyze the retrieved memories to find relevant information
2. Consider synonyms and related concepts
3. If memories mention specific dates/times, use those to answer time-related questions
4. If memories contain contradictory information, prioritize the most recent memory
5. Focus on the content of the memories, not just exact word matches

**For factual questions (What/When/Where/Who):**
- Answer based on direct information in the memories
- If the specific fact is not mentioned, respond: "Not answerable"

**For inference/reasoning questions (Would/Could/Likely):**
- You CAN make reasonable inferences based on related information in the memories

**When to say "Not answerable":**
- If the question asks about a specific person but the memories are about a DIFFERENT person
- If the question asks about an event/action that is NOT mentioned in ANY memories

Provide a concise, direct answer based on the available information, or state "Not answerable" if the specific information is not present."""

        try:
            return self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _clean_json_response(self, content: str) -> str:
        import re
        if not content:
            return content
        clean = content.strip()
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean).strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)
        try:
            json.loads(clean)
            return clean
        except Exception:
            return content

    def _retry_with_json_validation(self, prompt: str, max_retries: int = 10) -> str:
        import time
        for attempt in range(max_retries):
            try:
                temperature = 0.0 if attempt == 0 else 0.1
                response = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=16384,
                    response_format={"type": "json_object"},
                )
                if not response:
                    raise ValueError("Empty response from LLM")
                clean = self._clean_json_response(response)
                json.loads(clean)
                return clean
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(0.5 * (2 ** attempt), 16.0)
                    print(f"  LLM call failed (attempt {attempt + 1}/{max_retries}): "
                          f"{str(e)[:80]}, retrying in {wait_time:.1f}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"  LLM call failed after {max_retries} attempts: {str(e)[:100]}", flush=True)
                    raise

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_messages(self, messages: List[Dict]) -> str:
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        )

    def _get_existing_memories_for_routing(
        self,
        user_id: str,
        messages: Optional[List[Dict]] = None,
    ) -> Dict[str, str]:
        query_text = " ".join(m.get("content", "") for m in messages) if messages else ""
        sections: Dict[str, str] = {}
        for mem_type, store in self.vector_stores.items():
            results = store.search(query_text, top_k=MEMORY_CONSTRUCTION_TOP_K)
            memories = [
                r.get("memory", "") for r in results
                if isinstance(r, dict) and r.get("memory")
            ][:MEMORY_CONSTRUCTION_TOP_K]
            sections[mem_type] = "\n".join(memories) if memories else "[No existing memories]"
        return sections

    def _format_routed_memory(self, memory_obj: Dict, mem_type: str) -> str:
        title   = str(memory_obj.get("title",   "") or "").strip()
        summary = str(memory_obj.get("summary", "") or "").strip()
        details = str(memory_obj.get("details", "") or "").strip()
        time_val= str(memory_obj.get("time",    "") or "").strip()

        parts: List[str] = []
        if title and summary:
            parts.append(f"{title}: {summary}")
        elif summary:
            parts.append(summary)
        elif title:
            parts.append(title)
        if details:
            parts.append(f"Details: {details}" if parts else details)

        text = " | ".join(p for p in parts if p) or json.dumps(memory_obj, ensure_ascii=False)
        return f"{time_val}: {text}" if time_val else text


# ---------------------------------------------------------------------------
# MemorySystemWithGraph
# ---------------------------------------------------------------------------

class MemorySystemWithGraph(MemorySystemBase):
    """
    Extends MemorySystemBase with:
    - CrossReferenceGraph for cross-type memory links
    - _id_to_store_idx for O(1) memory lookup by hash ID
    - _fetch_memory_by_id and _score_memory_against_query used by eval runners
    - Extended save/load for graph and ID index persistence
    """

    def __init__(self, llm_client, embedding_func=None):
        super().__init__(llm_client, embedding_func)
        self.cross_ref_graph = CrossReferenceGraph()
        self._id_to_store_idx: Dict[str, Tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # ID-based memory access (used by eval runners for graph expansion)
    # ------------------------------------------------------------------

    def _fetch_memory_by_id(self, memory_id: str, expected_type: str) -> Optional[Dict]:
        """Retrieve a stored memory dict by its hash ID (O(1) via index)."""
        entry = self._id_to_store_idx.get(memory_id)
        if entry is None:
            store = self.vector_stores.get(expected_type)
            if store:
                for idx, meta in enumerate(store.metadata):
                    if isinstance(meta, dict) and meta.get("memory_id") == memory_id:
                        entry = (expected_type, idx)
                        self._id_to_store_idx[memory_id] = entry
                        break
        if entry is None:
            return None
        store_type, idx = entry
        store = self.vector_stores.get(store_type)
        if store is None or idx >= len(store.memories):
            return None
        meta = store.metadata[idx] if idx < len(store.metadata) else {}
        return {"memory": store.memories[idx], "score": None, "metadata": meta}

    def _score_memory_against_query(self, memory_id: str, mem_type: str, query_vec) -> float:
        """Return cosine similarity of a stored memory against a normalised query vector."""
        import numpy as np
        entry = self._id_to_store_idx.get(memory_id)
        if not entry:
            return 0.0
        store_type, idx = entry
        store = self.vector_stores.get(store_type)
        if store is None or not hasattr(store.index, "reconstruct"):
            return 0.0
        try:
            stored_vec = store.index.reconstruct(int(idx))
            return float(np.dot(query_vec[0], stored_vec))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Persistence (extends parent)
    # ------------------------------------------------------------------

    def save(self, db_path: str) -> None:
        super().save(db_path)
        self.cross_ref_graph.save(os.path.join(db_path, "cross_ref_graph.json"))
        serialisable = {k: list(v) for k, v in self._id_to_store_idx.items()}
        with open(os.path.join(db_path, "id_to_store_idx.json"), "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        print(f"  Cross-reference graph: {self.cross_ref_graph.stats()}")

    def load(self, db_path: str) -> bool:
        if not super().load(db_path):
            return False
        graph_path = os.path.join(db_path, "cross_ref_graph.json")
        if os.path.exists(graph_path):
            self.cross_ref_graph.load(graph_path)
            print(f"  Loaded cross-reference graph: {self.cross_ref_graph.stats()}")
        else:
            print("  Warning: no cross_ref_graph.json found — graph is empty")
        idx_path = os.path.join(db_path, "id_to_store_idx.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._id_to_store_idx = {k: tuple(v) for k, v in raw.items()}
            print(f"  Loaded ID index: {len(self._id_to_store_idx)} entries")
        else:
            print("  Warning: id_to_store_idx.json not found — rebuilding from metadata")
            self._rebuild_id_index()
        return True

    def _rebuild_id_index(self) -> None:
        self._id_to_store_idx = {}
        for mem_type, store in self.vector_stores.items():
            for idx, meta in enumerate(store.metadata):
                if isinstance(meta, dict):
                    mid = meta.get("memory_id")
                    if mid:
                        self._id_to_store_idx[mid] = (mem_type, idx)
        print(f"  Rebuilt ID index: {len(self._id_to_store_idx)} entries")


# ---------------------------------------------------------------------------
# MemorySystemFinal  (two-call extraction pipeline)
# ---------------------------------------------------------------------------

class MemorySystemFinal(MemorySystemWithGraph):
    """
    Two-call extraction pipeline:

    Call 1 — PROMPT_A_EXTRACT_RELATE_ROUTE
        Input : conversation timestamp + new messages
        Output: typed atoms + new-to-new links

    Call 1.5 (optional) — PROMPT_C_SELF_CHECK_EXTRACTION
        Input : atoms from Call 1 + original messages
        Output: additional atoms and links

    Call 2 — PROMPT_B_ASSIGN_OPS_LINKS
        Input : atoms + existing memories
        Output: ADD/UPDATE/SKIP per atom + new-to-existing links
    """

    def __init__(
        self,
        llm_client,
        embedding_func=None,
        self_check: bool = False,
        # legacy kwargs accepted and ignored
        prompt_version: str = "prompt_v1",
        extraction_prompt: str = "final",
    ):
        super().__init__(llm_client, embedding_func)
        self.self_check = self_check
        self.prompts["extract_relate_route"]  = PROMPT_A_EXTRACT_RELATE_ROUTE
        self.prompts["assign_ops_links"]      = PROMPT_B_ASSIGN_OPS_LINKS
        self.prompts["self_check_extraction"] = PROMPT_C_SELF_CHECK_EXTRACTION

    # ------------------------------------------------------------------
    # add() — two-call pipeline
    # ------------------------------------------------------------------

    def add(
        self,
        messages: List[Dict],
        user_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        operation_stats = {
            "semantic":   {"ADD": 0, "UPDATE": 0, "SKIP": 0},
            "episodic":   {"ADD": 0, "UPDATE": 0, "SKIP": 0},
            "procedural": {"ADD": 0, "UPDATE": 0, "SKIP": 0},
            "route": {
                "INVALID_MEM_TYPE": 0, "MISSING_MEM_OBJ": 0,
                "INVALID_MEM_FORMAT": 0, "INVALID_MEM_OP": 0,
            },
        }

        atoms, new_to_new_links = self._extract_relate_route(messages, metadata)
        if not atoms:
            return {"results": [], "memories_added": 0, "operation_stats": operation_stats}

        if self.self_check:
            atoms, new_to_new_links = self._self_check_extraction(
                messages, atoms, new_to_new_links, metadata
            )

        ops, existing_links = self._assign_ops_and_existing_links(
            atoms, user_id, messages, metadata
        )

        atom_id_to_info: Dict[int, Dict] = {}

        for op in ops:
            raw_atom_id = op.get("atom_id")
            action = str(op.get("action", "")).upper()
            if raw_atom_id is None or action not in {"ADD", "UPDATE", "SKIP"}:
                operation_stats["route"]["INVALID_MEM_OP"] += 1
                continue
            try:
                atom_id = int(raw_atom_id)
            except (TypeError, ValueError):
                operation_stats["route"]["INVALID_MEM_OP"] += 1
                continue
            if atom_id >= len(atoms):
                continue

            atom = atoms[atom_id]
            mem_type = str(atom.get("type", "")).strip().lower()
            if mem_type not in self.vector_stores:
                operation_stats["route"]["INVALID_MEM_TYPE"] += 1
                continue

            if action == "SKIP":
                existing_id = str(op.get("existing_id", "")).strip()
                atom_id_to_info[atom_id] = {
                    "action": "SKIP", "mem_type": mem_type, "memory_id": existing_id,
                }
                operation_stats[mem_type]["SKIP"] += 1
                continue

            memory_text = self._format_routed_memory(atom, mem_type)
            if not memory_text:
                operation_stats["route"]["INVALID_MEM_FORMAT"] += 1
                continue

            mem_id = make_memory_id(mem_type, memory_text)
            atom_id_to_info[atom_id] = {
                "action": action, "mem_type": mem_type,
                "memory_id": mem_id, "memory_text": memory_text,
            }
            operation_stats[mem_type][action] += 1

        by_type: Dict[str, Dict[str, List]] = {
            k: {"texts": [], "metadatas": []} for k in self.vector_stores
        }
        for info in atom_id_to_info.values():
            if info["action"] not in ("ADD", "UPDATE"):
                continue
            mem_type = info["mem_type"]
            by_type[mem_type]["texts"].append(info["memory_text"])
            by_type[mem_type]["metadatas"].append({
                "user_id": user_id, "type": "memory",
                "memory_type": mem_type, "action": info["action"],
                "memory_id": info["memory_id"],
            })

        for mem_type, bundle in by_type.items():
            if bundle["texts"]:
                store     = self.vector_stores[mem_type]
                start_idx = len(store.memories)
                store.add(bundle["texts"], bundle["metadatas"])
                for i, meta in enumerate(bundle["metadatas"]):
                    mid = meta.get("memory_id")
                    if mid:
                        self._id_to_store_idx[mid] = (mem_type, start_idx + i)

        results = [
            {
                "memory_type": info["mem_type"],
                "memory":      info["memory_text"],
                "user_id":     user_id,
                "memory_id":   info["memory_id"],
            }
            for info in atom_id_to_info.values()
            if info["action"] in ("ADD", "UPDATE")
        ]

        n_accepted = 0

        def _register(src_id: str, tgt_id: str, relation: str) -> bool:
            if not src_id or not tgt_id:
                return False
            if src_id not in self._id_to_store_idx or tgt_id not in self._id_to_store_idx:
                return False
            src_type = self._id_to_store_idx[src_id][0]
            tgt_type = self._id_to_store_idx[tgt_id][0]
            self.cross_ref_graph.add_link(src_id, tgt_id, relation, src_type, tgt_type)
            return True

        def _to_atom_id(val):
            if isinstance(val, list):
                val = val[0] if val else None
            if val is None:
                return None
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        for lk in new_to_new_links:
            src_atom = _to_atom_id(lk.get("source"))
            tgt_atom = _to_atom_id(lk.get("target"))
            relation = str(lk.get("relation", "")).strip()
            if src_atom is None or tgt_atom is None or not relation:
                continue
            src_info = atom_id_to_info.get(src_atom)
            tgt_info = atom_id_to_info.get(tgt_atom)
            if not src_info or not tgt_info:
                continue
            if _register(src_info["memory_id"], tgt_info["memory_id"], relation):
                n_accepted += 1

        for lk in existing_links:
            relation = str(lk.get("relation", "")).strip()
            if not relation:
                continue
            if "source_atom" in lk:
                src_info = atom_id_to_info.get(_to_atom_id(lk["source_atom"]))
                src_id   = src_info["memory_id"] if src_info else None
            elif "source_existing_id" in lk:
                src_id = str(lk["source_existing_id"]).strip()
            else:
                continue
            if "target_atom" in lk:
                tgt_info = atom_id_to_info.get(_to_atom_id(lk["target_atom"]))
                tgt_id   = tgt_info["memory_id"] if tgt_info else None
            elif "target_existing_id" in lk:
                tgt_id = str(lk["target_existing_id"]).strip()
            else:
                continue
            if _register(src_id, tgt_id, relation):
                n_accepted += 1

        total_links = len(new_to_new_links) + len(existing_links)
        if total_links:
            print(
                f"  Links: {len(new_to_new_links)} new-to-new + "
                f"{len(existing_links)} new-to-existing = "
                f"{total_links} proposed, {n_accepted} accepted"
            )

        return {"results": results, "memories_added": len(results), "operation_stats": operation_stats}

    # ------------------------------------------------------------------
    # Call 1: Extract → Relate → Route
    # ------------------------------------------------------------------

    def _extract_relate_route(
        self,
        messages: List[Dict],
        metadata: Optional[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        timestamp = metadata.get("timestamp", "Not provided") if metadata else "Not provided"
        prompt = Template(self.prompts["extract_relate_route"]).render(
            conversation_timestamp=timestamp,
            messages=self._format_messages(messages),
        )
        try:
            response = self._retry_with_json_validation(prompt)
            if not response:
                print("  [Call 1] empty response")
                return [], []
            result = json.loads(response)
            if not isinstance(result, dict):
                print("  [Call 1] response is not a JSON object")
                return [], []
            atoms = [a for a in result.get("atoms", []) if isinstance(a, dict) and "id" in a and "type" in a]
            links = [lk for lk in result.get("links", []) if isinstance(lk, dict) and all(k in lk for k in ("source", "target", "relation"))]
            print(f"  [Call 1] atoms={len(atoms)}, links={len(links)}")
            return atoms, links
        except Exception as e:
            print(f"  [Call 1] error: {str(e)[:100]}")
            return [], []

    # ------------------------------------------------------------------
    # Call 1.5 (optional): Self-Check Extraction
    # ------------------------------------------------------------------

    def _self_check_extraction(
        self,
        messages: List[Dict],
        atoms: List[Dict],
        links: List[Dict],
        metadata: Optional[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        next_id   = len(atoms)
        timestamp = metadata.get("timestamp", "Not provided") if metadata else "Not provided"
        prompt = Template(self.prompts["self_check_extraction"]).render(
            conversation_timestamp=timestamp,
            messages=self._format_messages(messages),
            existing_atoms_json=json.dumps(atoms, ensure_ascii=False, indent=2),
            next_id=next_id,
        )
        try:
            response = self._retry_with_json_validation(prompt)
            if not response:
                print("  [Call 1.5] empty response — skipping self-check")
                return atoms, links
            result = json.loads(response)
            if not isinstance(result, dict):
                return atoms, links

            additional_atoms = result.get("additional_atoms", [])
            additional_links = result.get("additional_links", [])
            if not isinstance(additional_atoms, list):
                additional_atoms = []
            if not isinstance(additional_links, list):
                additional_links = []

            valid_new_atoms: List[Dict] = []
            id_remap: Dict[int, int] = {}
            for atom in additional_atoms:
                if not isinstance(atom, dict) or "type" not in atom:
                    continue
                reported_id  = atom.get("id")
                canonical_id = next_id + len(valid_new_atoms)
                if reported_id is not None:
                    id_remap[int(reported_id)] = canonical_id
                atom = {**atom, "id": canonical_id}
                valid_new_atoms.append(atom)

            existing_ids = {a["id"] for a in atoms}
            valid_new_links: List[Dict] = []
            for lk in additional_links:
                if not isinstance(lk, dict) or not all(k in lk for k in ("source", "target", "relation")):
                    continue
                try:
                    src = int(lk["source"])
                    tgt = int(lk["target"])
                except (TypeError, ValueError):
                    continue
                src = id_remap.get(src, src) if src not in existing_ids else src
                tgt = id_remap.get(tgt, tgt) if tgt not in existing_ids else tgt
                valid_new_links.append({**lk, "source": src, "target": tgt})

            print(f"  [Call 1.5] additional_atoms={len(valid_new_atoms)}, additional_links={len(valid_new_links)}")
            return atoms + valid_new_atoms, links + valid_new_links
        except Exception as e:
            print(f"  [Call 1.5] error: {str(e)[:100]} — skipping self-check")
            return atoms, links

    # ------------------------------------------------------------------
    # Call 2: Assign Operations + Existing Links
    # ------------------------------------------------------------------

    def _assign_ops_and_existing_links(
        self,
        atoms: List[Dict],
        user_id: str,
        messages: List[Dict],
        metadata: Optional[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        timestamp = metadata.get("timestamp", "Not provided") if metadata else "Not provided"
        existing  = self._get_existing_memories_for_routing(user_id, messages)
        prompt = Template(self.prompts["assign_ops_links"]).render(
            conversation_timestamp=timestamp,
            existing_semantic_memories  =existing.get("semantic",   "[No existing memories]"),
            existing_episodic_memories  =existing.get("episodic",   "[No existing memories]"),
            existing_procedural_memories=existing.get("procedural", "[No existing memories]"),
            atoms_json=json.dumps(atoms, ensure_ascii=False, indent=2),
        )
        try:
            response = self._retry_with_json_validation(prompt)
            if not response:
                print("  [Call 2] empty response — defaulting all atoms to ADD")
                return [{"atom_id": a["id"], "action": "ADD"} for a in atoms], []
            result = json.loads(response)
            if not isinstance(result, dict):
                return [{"atom_id": a["id"], "action": "ADD"} for a in atoms], []

            ops = [op for op in result.get("operations", []) if isinstance(op, dict) and "atom_id" in op and "action" in op]
            existing_links = [
                lk for lk in result.get("existing_links", [])
                if isinstance(lk, dict)
                and ("source_atom" in lk or "source_existing_id" in lk)
                and ("target_atom" in lk or "target_existing_id" in lk)
                and "relation" in lk
            ]

            covered = {op["atom_id"] for op in ops}
            for atom in atoms:
                if atom["id"] not in covered:
                    ops.append({"atom_id": atom["id"], "action": "ADD"})

            adds    = sum(1 for op in ops if str(op.get("action", "")).upper() == "ADD")
            updates = sum(1 for op in ops if str(op.get("action", "")).upper() == "UPDATE")
            skips   = sum(1 for op in ops if str(op.get("action", "")).upper() == "SKIP")
            print(f"  [Call 2] ADD={adds}, UPDATE={updates}, SKIP={skips}, existing_links={len(existing_links)}")
            return ops, existing_links
        except Exception as e:
            print(f"  [Call 2] error: {str(e)[:100]} — defaulting all atoms to ADD")
            return [{"atom_id": a["id"], "action": "ADD"} for a in atoms], []
