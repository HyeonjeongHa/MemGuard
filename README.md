# MemGuard

MemGuard is a long-term memory system that uses a **two-call extraction pipeline** to build typed, relation-aware memories (semantic / episodic / procedural) from conversations, and retrieves them via cross-reference graph expansion.

This release includes two evaluation tracks:

| Track | Script | Datasets |
|---|---|---|
| **Utility** | `eval/runner.py` | LOCOMO · LongMemEval · PerLTQA |
| **Hallucination** | `HaluMem/eval/eval_memguard.py` | HaluMem-Medium |

---

## Repository Layout

```
MemGuard/
├── config.py                      # API keys, model defaults
├── llm_client.py                  # OpenAI-compatible LLM client
├── memguard_memory_system.py      # Core memory system (VectorStore → MemorySystemFinal)
├── prompts.py                     # All LLM prompts
├── eval/
│   ├── runner.py                  # Utility evaluation runner
│   ├── datasets.py                # Dataset loaders
│   ├── llm_judge.py               # LLM-based answer judge
│   └── metrics.py                 # Accuracy aggregation
├── HaluMem/
│   ├── data/
│   │   └── HaluMem-Medium.jsonl   ← download separately (see §2 below)
│   └── eval/
│       ├── eval_memguard.py       # Hallucination eval — memory builder + retriever
│       ├── evaluation.py          # Hallucination eval — metric aggregator
│       ├── eval_tools.py          # LLM judge prompts for hallucination metrics
│       ├── llms.py                # QA answer LLM client
│       └── prompts.py             # QA answer prompt
└── data/
    ├── locomo/locomo_conversations.json
    ├── longmemeval/longmemeval_s_cleaned.json   ← download separately (see §3 below)
    ├── longmemeval/splits/longmemeval_splits.json
    └── perltqa/perltqa_v2_standard.json
```

---

## Prerequisites

### 1. Install dependencies

```bash
pip install openai faiss-cpu jinja2 tqdm tenacity python-dotenv
```

> Use `faiss-gpu` instead of `faiss-cpu` if a GPU is available.

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."          # Required
export OPENAI_BASE_URL=""               # Optional: set for proxy or Azure endpoint
```

### 3. Download large datasets

Two dataset files are not included in the repository due to their size. Download them before running any evaluations.

**LongMemEval** (required for `--dataset longmemeval`):

```bash
# Install the Hugging Face datasets library if needed
pip install datasets huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="xiaowu0162/longmemeval-cleaned",
    filename="longmemeval_s_cleaned.json",
    repo_type="dataset",
)
os.makedirs("data/longmemeval", exist_ok=True)
shutil.copy(path, "data/longmemeval/longmemeval_s_cleaned.json")
print("Done:", "data/longmemeval/longmemeval_s_cleaned.json")
EOF
```

**HaluMem-Medium** (required for hallucination evaluation):

```bash
mkdir -p HaluMem/data
# Download HaluMem-Medium.jsonl from the HaluMem project page and place it at:
# HaluMem/data/HaluMem-Medium.jsonl
```

The default paths in `eval/runner.py` and `eval_memguard.py` already point to the above locations, so no extra flags are needed once the files are in place.

---

## 1. Utility Evaluation

Evaluates **question-answering accuracy** on LOCOMO, LongMemEval, and PerLTQA using MemGuard as the memory backend.

### 1.1 Quick start

```bash
# LOCOMO
python -m eval.runner \
  --dataset locomo \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --use_graph_expansion \
  --use_conf_aware_retrieval

# LongMemEval
python -m eval.runner \
  --dataset longmemeval \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --use_graph_expansion \
  --use_conf_aware_retrieval

# PerLTQA
python -m eval.runner \
  --dataset perltqa \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --use_graph_expansion \
  --use_conf_aware_retrieval
```

Results are saved under `logs/<dataset>/<model_name>/`.

> **Self-check extraction is on by default.** MemGuard runs an extra LLM pass after Call 1 to catch missed facts. Use `--no-self-check-extraction` to disable it.

---

### 1.2 Two-phase workflow (build then answer)

For large datasets, split memory building from answering to avoid re-processing on reruns.

**Phase 1 — Build memories**

```bash
python -m eval.runner \
  --dataset longmemeval \
  --mode build \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini
```

**Phase 2 — Answer questions** (loads saved memories)

```bash
python -m eval.runner \
  --dataset longmemeval \
  --mode answer \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --use_graph_expansion \
  --use_conf_aware_retrieval
```

---

### 1.3 LongMemEval with predefined splits

```bash
python -m eval.runner \
  --dataset longmemeval \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --split test \
  --use_graph_expansion \
  --use_conf_aware_retrieval
```

`--split` accepts `sft`, `rl`, or `test` (defined in `data/longmemeval/splits/longmemeval_splits.json`).

---

### 1.4 Parallel processing (LongMemEval)

```bash
python -m eval.runner \
  --dataset longmemeval \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --parallel \
  --workers 8 \
  --use_graph_expansion \
  --use_conf_aware_retrieval
```

---

### 1.5 Key options reference

| Option | Default | Description |
|---|---|---|
| `--dataset` | `locomo` | `locomo`, `longmemeval`, or `perltqa` |
| `--mode` | `full` | `full` (build + answer), `build`, or `answer` |
| `--model` | from config | Memory extractor model (e.g. `gpt-4.1-mini`) |
| `--model_name` | — | Short label used for output directory names |
| `--judge-model` | from config | LLM judge model (e.g. `gpt-4.1`) |
| `--api-key` | env `OPENAI_API_KEY` | Override API key |
| `--base-url` | env `OPENAI_BASE_URL` | Override API base URL |
| `--use_graph_expansion` | off | Enable BFS graph expansion during retrieval |
| `--use_conf_aware_retrieval` | off | LLM assigns per-type budget weights |
| `--compose` | off | Merge primary + expansion into one context entry |
| `--max-hops` | `1` | BFS depth for graph expansion |
| `--hop-decay` | `0.85` | Score decay per hop beyond the first |
| `--self-check-extraction` | **on** | Extra LLM pass after Call 1 to catch missed facts; disable with `--no-self-check-extraction` |
| `--top-k` | `10` | Number of memories retrieved per question |
| `--sessions` | all | Limit conversation sessions per sample |
| `--questions` | all | Limit questions evaluated |
| `--concurrency` | `1` | Parallel question-answering threads per sample |
| `--parallel` | off | Parallel sample processing (LongMemEval only) |
| `--workers` | `4` | Worker count for `--parallel` |
| `--output-dir` | `logs` | Base directory for result files |
| `--verbose` | off | Print retrieved memories and generated answers |
| `--split` | — | LongMemEval split: `sft`, `rl`, or `test` |
| `--db-path` | auto | Override FAISS database directory |

---

### 1.6 Output files

```
logs/<dataset>/<model_name>/<run_tag>/
├── summary.json              # Overall accuracy + per-conversation breakdown
├── results.json              # All individual results (LongMemEval)
└── <id>_results.json         # Per-conversation / per-sample results
```

---

## 2. Hallucination Evaluation

Evaluates **memory hallucination** — whether extracted memories are grounded, whether outdated facts are correctly updated, and whether retrieved context yields accurate QA answers.

> **Before running:** Download `HaluMem-Medium.jsonl` and place it at `HaluMem/data/HaluMem-Medium.jsonl` (see Prerequisites §3 above).

### 2.1 Step 1 — Build memories and generate responses

```bash
python HaluMem/eval/eval_memguard.py \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --version v1 \
  --use_graph_expansion \
  --use_conf_aware_routing \
  --top_k 20
```

This processes every user in `HaluMem/data/HaluMem-Medium.jsonl`, builds a per-user memory system, retrieves memories, generates answers, and writes results to:

```
HaluMem/eval/results/memguard-<model_label>-<routing_tag>-<version>/
└── memguard_eval_results.jsonl
```

> **Self-check extraction is on by default.** Use `--no-self_check` to disable it.

---

### 2.2 Step 2 — Aggregate hallucination metrics

```bash
python HaluMem/eval/evaluation.py \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --version v1 \
  --use_graph_expansion \
  --use_conf_aware_routing \
  --top_k 20
```

Pass the **same flags as Step 1** so the script resolves the correct results directory. It calls an LLM judge for each metric and writes:

```
HaluMem/eval/results/memguard-<model_label>-<routing_tag>-<version>/
└── memguard_eval_stat_result.json
```

---

### 2.3 Key options reference

#### eval_memguard.py

| Option | Default | Description |
|---|---|---|
| `--model` | — | **Required.** Model name (e.g. `gpt-4.1-mini`) |
| `--model_name` | — | Short label for the results directory |
| `--version` | `default` | Run label appended to the output directory |
| `--data_path` | `HaluMem/data/HaluMem-Medium.jsonl` | Path to HaluMem JSONL data |
| `--use_graph_expansion` | off | Enable BFS graph expansion during retrieval |
| `--use_conf_aware_routing` | off | LLM assigns per-type confidence weights |
| `--use_routed_memory` | off | Route query to primary memory type(s) |
| `--compose` | off | Merge primary + expansion into one context entry |
| `--max_hops` | `1` | BFS depth for graph expansion |
| `--hop_decay` | `0.85` | Score decay per hop beyond the first |
| `--self_check` | **on** | Extra LLM pass after Call 1; disable with `--no-self_check` |
| `--top_k` | `20` | Number of memories retrieved per query |
| `--max_workers` | `1` | Parallel user processing workers |

#### evaluation.py

| Option | Default | Description |
|---|---|---|
| `--model` | — | **Required.** Must match the `eval_memguard.py` run |
| `--model_name` | — | Must match the `eval_memguard.py` run |
| `--version` | `default` | Must match the `eval_memguard.py` run |
| `--user_num` | `20` | Maximum number of users to evaluate |
| `--max_workers` | `10` | Parallel LLM judge workers |
| (retrieval flags) | — | Same as `eval_memguard.py` — used for directory resolution |

---

### 2.4 Output metrics

`memguard_eval_stat_result.json` contains:

| Metric | Description |
|---|---|
| `memory_integrity.recall(all)` | Fraction of gold memory points captured |
| `memory_integrity.weighted_recall` | Importance-weighted recall |
| `memory_accuracy.target_accuracy(all)` | Precision of extracted memories against gold |
| `memory_accuracy.interference_accuracy` | Fraction of interference memories correctly rejected |
| `memory_extraction_f1` | F1 of integrity recall and target accuracy |
| `memory_update.correct_update_memory_ratio` | Fraction of fact updates that are correct |
| `memory_update.hallucination_update_memory_ratio` | Fraction that are hallucinated updates |
| `memory_update.omission_update_memory_ratio` | Fraction of missed updates |
| `question_answering.correct_qa_ratio` | Fraction of QA answers judged Correct |
| `question_answering.hallucination_qa_ratio` | Fraction judged Hallucination |
| `question_answering.omission_qa_ratio` | Fraction judged Omission |
| `memory_type_accuracy` | Per-type (Event / Persona / Relationship) accuracy |
| `time_consuming` | Wall-clock minutes for memory building and retrieval |

---

## Environment variable reference

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for all LLM and embedding calls |
| `OPENAI_BASE_URL` | Optional API base URL (proxy or Azure endpoint) |
| `OPENAI_MODEL` | Model for QA answer generation in hallucination eval (default: `gpt-4.1`) |
| `RETRY_TIMES` | LLM retry count in hallucination eval (default: `3`) |
| `WAIT_TIME_LOWER` | Min retry wait seconds in hallucination eval (default: `1`) |
| `WAIT_TIME_UPPER` | Max retry wait seconds in hallucination eval (default: `10`) |
| `OPENAI_MAX_TOKENS` | Optional max tokens override in hallucination eval |
| `OPENAI_TEMPERATURE` | Optional temperature override in hallucination eval |

Alternatively, create `HaluMem/eval/.env` with the above keys and they will be loaded automatically.

---

## Recommended full run (both tracks)

```bash
# Set credentials
export OPENAI_API_KEY="sk-..."

# --- Download large datasets (one-time) ---
# LongMemEval: see Prerequisites §3 for the hf_hub_download command
# HaluMem-Medium: place HaluMem-Medium.jsonl in HaluMem/data/ before continuing

# --- Utility evaluation (LOCOMO) ---
# Build memories once, then answer
python -m eval.runner \
  --dataset locomo \
  --mode build \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini

python -m eval.runner \
  --dataset locomo \
  --mode answer \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --use_graph_expansion \
  --use_conf_aware_retrieval

# --- Hallucination evaluation ---
# Step 1: build memories and generate responses
python HaluMem/eval/eval_memguard.py \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --version v1 \
  --use_graph_expansion \
  --use_conf_aware_routing

# Step 2: compute hallucination metrics
python HaluMem/eval/evaluation.py \
  --model gpt-4.1-mini \
  --model_name gpt-4.1-mini \
  --version v1 \
  --use_graph_expansion \
  --use_conf_aware_routing
```
