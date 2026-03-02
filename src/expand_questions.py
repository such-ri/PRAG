"""
Expand 300 questions to 1200 questions for inference testing.

For each original question, generates 3 additional variations:
  1. Yes/No question with the correct answer embedded (answer: yes)
  2. Yes/No question with a wrong answer embedded (answer: no)
  3. Statement confirmation with the correct answer (answer: yes)

Usage:
    python src/expand_questions.py

This creates a new folder `data_aug_1200_expanded/` with the same structure
as `data_aug_1200/`, but with 1200 questions per dataset instead of 300.
"""

import os
import re
import json
import random
from collections import defaultdict

# Support both Kaggle and local environments
try:
    from root_dir_path import ROOT_DIR
except ImportError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use actual repo root if ROOT_DIR doesn't contain data_aug_1200
if not os.path.exists(os.path.join(ROOT_DIR, "data_aug_1200")):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = os.path.join(ROOT_DIR, "data_aug_1200")
DST_DIR = os.path.join(ROOT_DIR, "data_aug_1200_expanded")

YES_ANSWERS = ["yes", "Yes", "correct", "right", "true", "True"]
NO_ANSWERS = ["no", "No", "incorrect", "wrong", "false", "False"]


def get_main_answer(answer):
    """Extract the main (first) answer string from answer field."""
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return answer


def get_wrong_answer(answer, all_answers):
    """Get a plausible wrong answer from the pool of all answers in the dataset."""
    main_ans = get_main_answer(answer).lower().strip()
    candidates = []
    for ans in all_answers:
        main_other = get_main_answer(ans).lower().strip()
        if main_other and main_other != main_ans and main_other not in ("yes", "no"):
            candidates.append(get_main_answer(ans))
    if candidates:
        return random.choice(candidates)
    return "unknown"


def clean_question(q):
    """Clean question string: ensure it ends with '?'."""
    q = q.strip()
    if not q.endswith("?"):
        q = q + "?"
    return q


def generate_variations(question, answer, wrong_answer):
    """
    Generate 3 question variations for a given question-answer pair.

    Returns list of (variant_question, variant_answer, variant_type) tuples.
    """
    main_ans = get_main_answer(answer)
    q = clean_question(question)

    # Check if the original answer is yes/no type
    is_yesno = get_main_answer(answer).lower().strip() in ("yes", "no")

    if is_yesno:
        # For yes/no questions, generate variations differently
        original_is_yes = get_main_answer(answer).lower().strip() == "yes"

        # V1: Affirmative rephrasing - "Is it true that [question]?"
        v1_q = f"Is the correct answer to the question \"{q}\" {main_ans}?"
        v1_a = YES_ANSWERS[:]

        # V2: Negation - "Is the opposite answer correct?"
        opposite = "no" if original_is_yes else "yes"
        v2_q = f"Is the correct answer to the question \"{q}\" {opposite}?"
        v2_a = NO_ANSWERS[:]

        # V3: Statement confirmation
        v3_q = f"For the question \"{q}\", the answer is {main_ans}, right?"
        v3_a = YES_ANSWERS[:]
    else:
        # For entity-answer questions
        # V1: Yes/No with correct answer
        v1_q = f"Is {main_ans} the correct answer to the following question: {q}"
        v1_a = YES_ANSWERS[:]

        # V2: Yes/No with wrong answer
        v2_q = f"Is {wrong_answer} the correct answer to the following question: {q}"
        v2_a = NO_ANSWERS[:]

        # V3: Statement confirmation
        v3_q = f"The answer to \"{q}\" is {main_ans}, correct?"
        v3_a = YES_ANSWERS[:]

    return [
        (v1_q, v1_a, "yes_correct"),
        (v2_q, v2_a, "no_wrong"),
        (v3_q, v3_a, "confirm"),
    ]


def load_full_data_for_total(dataset_dir, total_data):
    """
    For datasets with type-specific files (hotpotqa, 2wikimultihopqa),
    load full data (with passages/augment) by merging from type-specific files.
    """
    files = os.listdir(dataset_dir)
    type_files = [f for f in files if f != "total.json"]

    if not type_files:
        return total_data

    # Load all type-specific files
    all_type_data = {}
    for filename in type_files:
        with open(os.path.join(dataset_dir, filename), "r") as f:
            all_type_data[filename] = json.load(f)

    # Merge: for each total entry, find corresponding entry in type-specific file
    idx = {filename: 0 for filename in type_files}
    merged = []
    for data in total_data:
        typ = data.get("type")
        if typ is None:
            merged.append(data)
            continue
        type_file = typ + ".json"
        if type_file not in all_type_data:
            merged.append(data)
            continue
        if idx[type_file] >= len(all_type_data[type_file]):
            merged.append(data)
            continue
        aim_data = all_type_data[type_file][idx[type_file]]
        if aim_data["question"] == data["question"]:
            idx[type_file] += 1
            merged.append(aim_data)
        else:
            merged.append(data)

    return merged


def expand_dataset(dataset_dir, dataset_name):
    """Expand a single dataset from 300 to 1200 questions."""
    print(f"\n=== Processing {dataset_name} ===")

    # Load total.json
    total_path = os.path.join(dataset_dir, "total.json")
    with open(total_path, "r") as f:
        total_data = json.load(f)

    print(f"  Original questions: {len(total_data)}")

    # Try to get full data with passages/augment from type-specific files
    full_data = load_full_data_for_total(dataset_dir, total_data)

    # Collect all answers for wrong answer generation
    all_answers = [d["answer"] for d in full_data]

    # Set random seed for reproducibility
    random.seed(42)

    # Generate expanded data
    expanded = []
    for orig_idx, data in enumerate(full_data):
        question = data["question"]
        answer = data["answer"]

        # Add original question (test_id = orig_idx * 4)
        orig_entry = dict(data)
        orig_entry["test_id"] = orig_idx * 4
        orig_entry["original_test_id"] = orig_idx
        orig_entry["variant_type"] = "original"
        # Ensure passages field exists
        if "passages" not in orig_entry:
            if "golden_passages" in orig_entry:
                orig_entry["passages"] = orig_entry["golden_passages"]
            else:
                orig_entry["passages"] = []
        expanded.append(orig_entry)

        # Generate wrong answer
        wrong_ans = get_wrong_answer(answer, all_answers)

        # Generate 3 variations
        variations = generate_variations(question, answer, wrong_ans)
        for var_idx, (var_q, var_a, var_type) in enumerate(variations):
            var_entry = {
                "test_id": orig_idx * 4 + var_idx + 1,
                "original_test_id": orig_idx,
                "question": var_q,
                "answer": var_a,
                "variant_type": var_type,
                "original_question": question,
                "original_answer": answer,
            }
            # Copy passages from original
            if "passages" in data:
                var_entry["passages"] = data["passages"]
            elif "golden_passages" in data:
                var_entry["passages"] = data["golden_passages"]
            else:
                var_entry["passages"] = []
            # Copy type if exists
            if "type" in data:
                var_entry["type"] = data["type"]
            # Copy qid if exists
            if "qid" in data:
                var_entry["qid"] = data["qid"]
            # Copy augment if exists (for encoding compatibility)
            if "augment" in data:
                var_entry["augment"] = data["augment"]
            expanded.append(var_entry)

    print(f"  Expanded questions: {len(expanded)}")
    return expanded


def expand_type_file(filepath, all_answers_pool):
    """Expand a type-specific file (bridge.json, comparison.json, etc.)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    random.seed(42)
    expanded = []
    for orig_idx, entry in enumerate(data):
        question = entry["question"]
        answer = entry["answer"]

        # Add original
        orig_entry = dict(entry)
        orig_entry["test_id"] = orig_idx * 4
        orig_entry["original_test_id"] = orig_idx
        orig_entry["variant_type"] = "original"
        if "passages" not in orig_entry:
            if "golden_passages" in orig_entry:
                orig_entry["passages"] = orig_entry["golden_passages"]
            else:
                orig_entry["passages"] = []
        expanded.append(orig_entry)

        # Generate wrong answer
        wrong_ans = get_wrong_answer(answer, all_answers_pool)

        # Generate variations
        variations = generate_variations(question, answer, wrong_ans)
        for var_idx, (var_q, var_a, var_type) in enumerate(variations):
            var_entry = {
                "test_id": orig_idx * 4 + var_idx + 1,
                "original_test_id": orig_idx,
                "question": var_q,
                "answer": var_a,
                "variant_type": var_type,
                "original_question": question,
                "original_answer": answer,
            }
            if "passages" in entry:
                var_entry["passages"] = entry["passages"]
            elif "golden_passages" in entry:
                var_entry["passages"] = entry["golden_passages"]
            else:
                var_entry["passages"] = []
            if "type" in entry:
                var_entry["type"] = entry["type"]
            if "qid" in entry:
                var_entry["qid"] = entry["qid"]
            if "augment" in entry:
                var_entry["augment"] = entry["augment"]
            expanded.append(var_entry)
    return expanded


def main():
    datasets = ["hotpotqa", "2wikimultihopqa", "popqa", "complexwebquestions"]

    for dataset in datasets:
        # Find model subdirectories
        dataset_base = os.path.join(SRC_DIR, dataset)
        if not os.path.exists(dataset_base):
            print(f"Skipping {dataset}: directory not found")
            continue

        for model_name in os.listdir(dataset_base):
            model_dir = os.path.join(dataset_base, model_name)
            if not os.path.isdir(model_dir):
                continue

            # Create output directory
            out_dir = os.path.join(DST_DIR, dataset, model_name)
            os.makedirs(out_dir, exist_ok=True)

            # Expand total.json (main file for inference)
            expanded_total = expand_dataset(model_dir, f"{dataset}/{model_name}")

            # Write expanded total.json
            out_total = os.path.join(out_dir, "total.json")
            with open(out_total, "w") as f:
                json.dump(expanded_total, f, indent=2, ensure_ascii=False)
            print(f"  Written: {out_total}")

            # Collect answer pool for type-specific files
            all_answers = [d["answer"] for d in expanded_total
                           if d.get("variant_type") == "original"]

            # Expand type-specific files
            files = os.listdir(model_dir)
            for filename in files:
                if filename == "total.json":
                    continue
                filepath = os.path.join(model_dir, filename)
                if not filename.endswith(".json"):
                    continue

                print(f"  Expanding {filename}...")
                expanded_type = expand_type_file(filepath, all_answers)
                out_path = os.path.join(out_dir, filename)
                with open(out_path, "w") as f:
                    json.dump(expanded_type, f, indent=2, ensure_ascii=False)
                print(f"    {len(expanded_type)} entries -> {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPANSION SUMMARY")
    print("=" * 60)
    for dataset in datasets:
        dataset_dir = os.path.join(DST_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue
        for model_name in os.listdir(dataset_dir):
            model_dir = os.path.join(dataset_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            print(f"\n{dataset}/{model_name}:")
            for filename in sorted(os.listdir(model_dir)):
                filepath = os.path.join(model_dir, filename)
                with open(filepath) as f:
                    data = json.load(f)
                print(f"  {filename}: {len(data)} entries")

    # Show first 10 expanded questions from each dataset for verification
    print("\n" + "=" * 60)
    print("SAMPLE EXPANDED QUESTIONS (first 8 from each dataset)")
    print("=" * 60)
    for dataset in datasets:
        dataset_dir = os.path.join(DST_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue
        for model_name in os.listdir(dataset_dir):
            total_path = os.path.join(dataset_dir, model_name, "total.json")
            with open(total_path) as f:
                data = json.load(f)
            print(f"\n--- {dataset}/{model_name} ---")
            for d in data[:8]:
                ans_display = str(d["answer"])[:50]
                print(f"  [{d['variant_type']:12s}] Q: {d['question'][:100]}")
                print(f"                A: {ans_display}")


if __name__ == "__main__":
    main()
