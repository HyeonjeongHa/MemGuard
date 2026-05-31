"""
HaluMem evaluation aggregator for MemGuard.

Run this after eval_memguard.py has generated JSONL results to compute
memory integrity, memory accuracy, memory update, and QA metrics.

Usage
-----
python HaluMem/eval/evaluation.py \\
  --model gpt-4.1-mini \\
  --version v1 \\
  --use_graph_expansion \\
  --use_conf_aware_routing \\
  --top_k 20
"""

import os
import json
import time
import copy
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from eval_tools import (
    evaluation_for_memory_accuracy,
    evaluation_for_memory_integrity,
    evaluation_for_question,
    evaluation_for_update_memory,
)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def process_user(idx: int, user_data: dict, max_workers: int = 10):
    uuid      = user_data["uuid"]
    user_name = user_data["user_name"]

    eval_results = {
        "memory_integrity_records":  [],
        "memory_accuracy_records":   [],
        "memory_update_records":     [],
        "question_answering_records":[],
    }

    memory_integrity_inputs  = []
    memory_accuracy_inputs   = []
    memory_update_inputs     = []
    question_answering_inputs= []

    for sid, session in enumerate(user_data["sessions"]):
        if session.get("is_generated_qa_session", False):
            continue

        golden_memories  = session["memory_points"]
        extract_memories = session["extracted_memories"]
        extract_str      = "\n".join(extract_memories)

        for memory in golden_memories:
            if memory["is_update"] == "True" and memory.get("memories_from_system", []):
                update_mem = copy.deepcopy(memory)
                update_mem["uuid"]       = uuid
                update_mem["ssession_id"]= sid
                memory_update_inputs.append(update_mem)
            else:
                mem = copy.deepcopy(memory)
                mem["uuid"]       = uuid
                mem["ssession_id"]= sid
                memory_integrity_inputs.append((mem, extract_str))

        dialogue_str = []
        for turn in session["dialogue"]:
            dialogue_str.append(f'[{turn["timestamp"]}]{turn["role"]}: {turn["content"]}')
            if turn["role"] == "assistant":
                dialogue_str.append("")
        dialogue_str = "\n".join(dialogue_str)

        golden_memories_str = "\n".join(
            m["memory_content"] for m in golden_memories if m["memory_source"] != "interference"
        )

        for memory in extract_memories:
            memory_accuracy_inputs.append((
                dialogue_str,
                golden_memories_str,
                {"uuid": uuid, "ssession_id": sid, "memory_content": memory},
            ))

        if "questions" in session:
            for qa in session["questions"]:
                new_qa = copy.deepcopy(qa)
                new_qa["uuid"]       = uuid
                new_qa["ssession_id"]= sid
                question_answering_inputs.append(new_qa)

    # Memory Integrity
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for memory, extract_str in memory_integrity_inputs:
            if not extract_str.strip():
                memory["memory_integrity_score"] = 0
                eval_results["memory_integrity_records"].append(memory)
                continue
            future = executor.submit(evaluation_for_memory_integrity, extract_str, memory["memory_content"])
            futures[future] = memory
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Integrity ([{idx}]{user_name})"):
            memory = futures[future]
            try:
                score = int(future.result().get("score"))
            except Exception:
                score = None
            memory["memory_integrity_score"] = score
            eval_results["memory_integrity_records"].append(memory)

    # Memory Accuracy
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dialogue_str, golden_str, memory in memory_accuracy_inputs:
            future = executor.submit(evaluation_for_memory_accuracy, dialogue_str, golden_str, memory["memory_content"])
            futures[future] = memory
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Accuracy ([{idx}]{user_name})"):
            memory = futures[future]
            try:
                result = future.result()
                score = int(result.get("accuracy_score"))
                included = result.get("is_included_in_golden_memories", "false")
            except Exception:
                score    = None
                included = "false"
            memory["memory_accuracy_score"]           = score
            memory["is_included_in_golden_memories"]  = included
            eval_results["memory_accuracy_records"].append(memory)

    # Memory Update
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for update_memory in memory_update_inputs:
            future = executor.submit(
                evaluation_for_update_memory,
                "\n".join(update_memory["memories_from_system"]),
                update_memory["memory_content"],
                "\n".join(update_memory["original_memories"]),
            )
            futures[future] = update_memory
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Memory Update ([{idx}]{user_name})"):
            update_memory = futures[future]
            try:
                update_type = future.result().get("evaluation_result")
            except Exception:
                update_type = None
            update_memory["memory_update_type"] = update_type
            eval_results["memory_update_records"].append(update_memory)

    # Question Answering
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for qa in question_answering_inputs:
            future = executor.submit(
                evaluation_for_question,
                qa["question"],
                qa["answer"],
                "\n".join(i["memory_content"] for i in qa["evidence"]),
                qa["system_response"],
            )
            futures[future] = qa
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Question Answering ([{idx}]{user_name})"):
            qa = futures[future]
            try:
                result_type = future.result().get("evaluation_result")
            except Exception:
                result_type = None
            qa["result_type"] = result_type
            eval_results["question_answering_records"].append(qa)

    return eval_results


def aggregate_eval_results(eval_results):
    # Memory Integrity
    mi_scores = mi_weighted_scores = mi_valid_num = mi_num = 0
    mi_weighted_valid_num = mi_weighted_num = 0
    int_scores = int_valid_num = int_num = 0

    for item in eval_results["memory_integrity_records"]:
        item["is_valid"] = True
        if item["memory_source"] != "interference":
            mi_num += 1
            mi_weighted_num += item["importance"]
        else:
            int_num += 1
        if item["memory_integrity_score"] is None:
            item["is_valid"] = False
            continue
        if item["memory_source"] != "interference":
            if item["memory_integrity_score"] == 2:
                mi_scores += 1
            mi_weighted_scores += 0.5 * item["memory_integrity_score"] * item["importance"]
            mi_valid_num += 1
            mi_weighted_valid_num += item["importance"]
        else:
            if item["memory_integrity_score"] == 0:
                int_scores += 1
            int_valid_num += 1

    s = eval_results["overall_score"]
    s["memory_integrity"]["recall(all)"]                  = mi_scores / mi_num
    s["memory_integrity"]["recall(valid)"]                = mi_scores / mi_valid_num
    s["memory_integrity"]["weighted_recall(all)"]         = mi_weighted_scores / mi_weighted_num
    s["memory_integrity"]["weighted_recall(valid)"]       = mi_weighted_scores / mi_weighted_valid_num
    s["memory_integrity"]["memory_valid_importance_sum"]  = mi_weighted_valid_num
    s["memory_integrity"]["memory_importance_sum"]        = mi_weighted_num
    s["memory_integrity"]["memory_valid_num"]             = mi_valid_num
    s["memory_integrity"]["memory_num"]                   = mi_num
    s["memory_accuracy"]["interference_accuracy(all)"]    = int_scores / int_num
    s["memory_accuracy"]["interference_accuracy(valid)"]  = int_scores / int_valid_num
    s["memory_accuracy"]["interference_memory_valid_num"] = int_valid_num
    s["memory_accuracy"]["interference_memory_num"]       = int_num

    # Memory Accuracy
    tgt_scores = tgt_valid_num = tgt_num = 0
    all_scores = all_valid_num = all_num = 0

    for item in eval_results["memory_accuracy_records"]:
        item["is_valid"] = True
        all_num += 1
        included = item["is_included_in_golden_memories"] in ["true", "True"]
        if included:
            tgt_num += 1
        if item["memory_accuracy_score"] is None:
            item["is_valid"] = False
            continue
        if included:
            tgt_scores += 0.5 * item["memory_accuracy_score"]
            tgt_valid_num += 1
        all_scores += 0.5 * item["memory_accuracy_score"]
        all_valid_num += 1

    s["memory_accuracy"]["target_accuracy(all)"]     = tgt_scores / tgt_num
    s["memory_accuracy"]["target_accuracy(valid)"]   = tgt_scores / tgt_valid_num
    s["memory_accuracy"]["target_memory_valid_num"]  = tgt_valid_num
    s["memory_accuracy"]["target_memory_num"]        = tgt_num
    s["memory_accuracy"]["weighted_accuracy(all)"]   = all_scores / all_num
    s["memory_accuracy"]["weighted_accuracy(valid)"] = all_scores / all_valid_num
    s["memory_accuracy"]["memory_valid_num"]         = all_valid_num
    s["memory_accuracy"]["memory_num"]               = all_num

    s["memory_extraction_f1"] = compute_f1(
        precision=s["memory_accuracy"]["target_accuracy(all)"],
        recall   =s["memory_integrity"]["recall(all)"],
    )

    # Memory Update
    correct_u = halluc_u = omit_u = other_u = update_num = update_valid = 0
    for item in eval_results["memory_update_records"]:
        item["is_valid"] = True
        update_num += 1
        ut = item["memory_update_type"]
        if ut not in ["Correct", "Hallucination", "Omission", "Other"]:
            item["is_valid"] = False
            continue
        update_valid += 1
        if ut == "Correct":      correct_u += 1
        elif ut == "Hallucination": halluc_u += 1
        elif ut == "Omission":   omit_u += 1
        elif ut == "Other":      other_u += 1

    s["memory_update"]["correct_update_memory_ratio(all)"]        = correct_u / update_num
    s["memory_update"]["correct_update_memory_ratio(valid)"]      = correct_u / update_valid
    s["memory_update"]["hallucination_update_memory_ratio(all)"]  = halluc_u / update_num
    s["memory_update"]["hallucination_update_memory_ratio(valid)"]= halluc_u / update_valid
    s["memory_update"]["omission_update_memory_ratio(all)"]       = omit_u / update_num
    s["memory_update"]["omission_update_memory_ratio(valid)"]     = omit_u / update_valid
    s["memory_update"]["other_update_memory_ratio(all)"]          = other_u / update_num
    s["memory_update"]["other_update_memory_ratio(valid)"]        = other_u / update_valid
    s["memory_update"]["update_memory_valid_num"]                 = update_valid
    s["memory_update"]["update_memory_num"]                       = update_num

    # Question Answering
    correct_q = halluc_q = omit_q = qa_num = qa_valid = 0
    for item in eval_results["question_answering_records"]:
        item["is_valid"] = True
        qa_num += 1
        rt = item["result_type"]
        if rt not in ["Correct", "Hallucination", "Omission"]:
            item["is_valid"] = False
            continue
        qa_valid += 1
        if rt == "Correct":         correct_q += 1
        elif rt == "Hallucination": halluc_q += 1
        elif rt == "Omission":      omit_q += 1

    s["question_answering"]["correct_qa_ratio(all)"]        = correct_q / qa_num
    s["question_answering"]["correct_qa_ratio(valid)"]      = correct_q / qa_valid
    s["question_answering"]["hallucination_qa_ratio(all)"]  = halluc_q / qa_num
    s["question_answering"]["hallucination_qa_ratio(valid)"]= halluc_q / qa_valid
    s["question_answering"]["omission_qa_ratio(all)"]       = omit_q / qa_num
    s["question_answering"]["omission_qa_ratio(valid)"]     = omit_q / qa_valid
    s["question_answering"]["qa_valid_num"]                 = qa_valid
    s["question_answering"]["qa_num"]                       = qa_num

    # Memory Type Accuracy
    for item in eval_results["memory_integrity_records"]:
        if "memory_integrity_score" not in item or "importance" not in item:
            continue
        score = 1 if item["memory_integrity_score"] == 2 else 0
        t = item["memory_type"]
        s["memory_type_accuracy"][t]["memory_integrity_acc"] += score
        s["memory_type_accuracy"][t]["total_num"] += 1

    for item in eval_results["memory_update_records"]:
        if "memory_update_type" not in item or "importance" not in item:
            continue
        score = 1 if item["memory_update_type"] == "Correct" else 0
        t = item["memory_type"]
        s["memory_type_accuracy"][t]["memory_update_acc"] += score
        s["memory_type_accuracy"][t]["total_num"] += 1

    for key in s["memory_type_accuracy"]:
        total = s["memory_type_accuracy"][key]["total_num"]
        if total > 0:
            s["memory_type_accuracy"][key]["memory_integrity_acc"] /= total
            s["memory_type_accuracy"][key]["memory_update_acc"]    /= total
            s["memory_type_accuracy"][key]["memory_acc"] = (
                s["memory_type_accuracy"][key]["memory_integrity_acc"]
                + s["memory_type_accuracy"][key]["memory_update_acc"]
            )

    return eval_results


def iter_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(
    model: str,
    model_name: str = None,
    version: str = "default",
    use_routed_memory: bool = True,
    use_graph_expansion: bool = True,
    use_conf_aware_routing: bool = False,
    top_k: int = 20,
    max_hops: int = 1,
    hop_decay: float = 0.85,
    compose: bool = False,
    user_num: int = 20,
    max_workers: int = 10,
):
    frame       = "memguard"
    model_label = model_name or (model.split("/")[-1] if "/" in model else model)
    compose_tag = "_composed" if compose else ""
    routing_tag = (
        f"routed-{use_routed_memory}_graph-{use_graph_expansion}"
        f"_conf-{use_conf_aware_routing}{compose_tag}_topk{top_k}"
        f"_hops{max_hops}_decay{hop_decay}"
    )
    dir_path    = f"results/{frame}-{model_label}-{routing_tag}-{version}/"
    data_path   = f"{dir_path}{frame}_eval_results.jsonl"
    output_file = os.path.join(dir_path, f"{frame}_eval_stat_result.json")
    tmp_dir     = os.path.join(dir_path, "tmp2")
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    for idx, user_data in enumerate(iter_jsonl(data_path), 1):
        uuid     = user_data["uuid"]
        tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

        if os.path.exists(tmp_file):
            print(f"⚡ Skipping user {uuid} ({idx}) — cached result found.")
        else:
            print(f"Processing user {uuid} ({idx})...")
            user_result = process_user(idx, user_data, max_workers)
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(user_result, f, ensure_ascii=False, indent=4)
            elapsed = time.time() - start_time
            print(f"✅ Finished user {uuid} ({idx}), elapsed {elapsed:.2f}s.")

        if idx >= user_num:
            break

    # Timing stats
    add_duration    = 0.0
    search_duration = 0.0
    for user_data in iter_jsonl(data_path):
        for session in user_data["sessions"]:
            add_duration    += session.get("add_dialogue_duration_ms", 0)
            for q in session.get("questions", []):
                search_duration += q.get("search_duration_ms", 0)

    add_duration    /= 1000 * 60
    search_duration /= 1000 * 60

    print("\n🔄 Aggregating all user results...")

    eval_results = {
        "overall_score": {
            "memory_integrity":  {},
            "memory_accuracy":   {},
            "memory_extraction_f1": 0,
            "memory_update":     {},
            "question_answering":{},
            "memory_type_accuracy": {
                "Event Memory":        {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
                "Persona Memory":      {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
                "Relationship Memory": {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
            },
            "time_consuming": {
                "add_dialogue_duration_time":  add_duration,
                "search_memory_duration_time": search_duration,
                "total_duration_time":         add_duration + search_duration,
            },
        },
        "memory_integrity_records":   [],
        "memory_accuracy_records":    [],
        "memory_update_records":      [],
        "question_answering_records": [],
    }

    for file_name in os.listdir(tmp_dir):
        if not file_name.endswith(".json"):
            continue
        with open(os.path.join(tmp_dir, file_name), "r", encoding="utf-8") as f:
            user_result = json.load(f)
        eval_results["memory_accuracy_records"].extend(user_result.get("memory_accuracy_records", []))
        eval_results["memory_integrity_records"].extend(user_result.get("memory_integrity_records", []))
        eval_results["memory_update_records"].extend(user_result.get("memory_update_records", []))
        eval_results["question_answering_records"].extend(user_result.get("question_answering_records", []))

    eval_results = aggregate_eval_results(eval_results)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    elapsed = time.time() - start_time
    print(f"✅ All done in {elapsed:.2f}s. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HaluMem evaluation aggregator for MemGuard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str, required=True,
                        help="Model name used during eval_memguard.py run.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short model label (for directory resolution).")
    parser.add_argument("--version",    type=str, default="default",
                        help="Version label matching the eval_memguard.py run.")
    parser.add_argument("--use_routed_memory",    action="store_true", default=True)
    parser.add_argument("--use_graph_expansion",  action="store_true", default=True)
    parser.add_argument("--use_conf_aware_routing", action="store_true", default=False)
    parser.add_argument("--top_k",     type=int,   default=20)
    parser.add_argument("--max_hops",  type=int,   default=1)
    parser.add_argument("--hop_decay", type=float, default=0.85)
    parser.add_argument("--compose",   action="store_true", default=False)
    parser.add_argument("--user_num",  type=int, default=20,
                        help="Maximum number of users to evaluate.")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Parallel workers for LLM evaluation calls.")
    args = parser.parse_args()

    main(
        model               =args.model,
        model_name          =args.model_name,
        version             =args.version,
        use_routed_memory   =args.use_routed_memory,
        use_graph_expansion =args.use_graph_expansion,
        use_conf_aware_routing=args.use_conf_aware_routing,
        top_k               =args.top_k,
        max_hops            =args.max_hops,
        hop_decay           =args.hop_decay,
        compose             =args.compose,
        user_num            =args.user_num,
        max_workers         =args.max_workers,
    )
