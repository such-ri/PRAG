"""
Expand 300 questions to 1200 questions for inference testing.

For each original question, generates 3 additional variations using a diverse
pool of question reformulation strategies. The variations include yes/no questions,
true/false questions, confirmation requests, negated assertions, choice questions,
and other flexible question forms.

Usage:
    python src/expand_questions.py

This creates a new folder `data_aug_1200_expanded/` with the same structure
as `data_aug_1200/`, but with 1200 questions per dataset instead of 300.
"""

import os
import json
import random

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

# Comprehensive answer sets - models may respond in various forms
YES_ANSWERS = ["yes", "Yes", "YES", "correct", "Correct", "right", "Right",
               "true", "True", "TRUE", "yeah", "Yeah", "certainly", "Certainly",
               "indeed", "Indeed", "absolutely", "Absolutely", "affirmative"]
NO_ANSWERS = ["no", "No", "NO", "incorrect", "Incorrect", "wrong", "Wrong",
              "false", "False", "FALSE", "not correct", "not right", "nope", "Nope"]
TRUE_ANSWERS = ["true", "True", "TRUE", "correct", "Correct", "yes", "Yes"]
FALSE_ANSWERS = ["false", "False", "FALSE", "incorrect", "Incorrect", "no", "No",
                 "wrong", "Wrong"]


def get_main_answer(answer):
    """Extract the main (first) answer string from answer field."""
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return answer


def get_all_answers_as_list(answer):
    """Return answer as a list regardless of input type."""
    if isinstance(answer, list):
        return answer
    return [answer]


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


def get_second_wrong_answer(answer, wrong1, all_answers):
    """Get a second distinct wrong answer."""
    main_ans = get_main_answer(answer).lower().strip()
    w1 = wrong1.lower().strip()
    candidates = []
    for ans in all_answers:
        main_other = get_main_answer(ans).lower().strip()
        if (main_other and main_other != main_ans and
                main_other != w1 and main_other not in ("yes", "no")):
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


def strip_question_mark(q):
    """Remove trailing question mark."""
    q = q.strip()
    if q.endswith("?"):
        q = q[:-1].strip()
    return q


# ============================================================================
# Diverse template pools for ENTITY-ANSWER questions
# Each function returns (question_text, answer_list, variant_type_label)
# ============================================================================

def entity_yesno_correct_v1(q, ans, wrong, wrong2):
    """Is [answer] the answer to [question]?"""
    return (f"Is {ans} the answer to: {q}",
            YES_ANSWERS[:], "yesno_correct")

def entity_yesno_correct_v2(q, ans, wrong, wrong2):
    """Is it true that [question] -> [answer]?"""
    return (f"Is it true that the answer to \"{strip_question_mark(q)}\" is {ans}?",
            YES_ANSWERS[:], "true_confirm")

def entity_yesno_correct_v3(q, ans, wrong, wrong2):
    """I believe ... Am I correct?"""
    return (f"I believe the answer to \"{q}\" is {ans}. Am I correct?",
            YES_ANSWERS[:], "belief_confirm")

def entity_yesno_correct_v4(q, ans, wrong, wrong2):
    """Can you confirm: ...?"""
    return (f"Can you confirm that the answer to \"{q}\" is {ans}?",
            YES_ANSWERS[:], "confirm_request")

def entity_yesno_correct_v5(q, ans, wrong, wrong2):
    """If asked [q], should one answer [ans]?"""
    return (f"If someone asks \"{q}\", should the answer be {ans}?",
            YES_ANSWERS[:], "should_answer")

def entity_yesno_correct_v6(q, ans, wrong, wrong2):
    """[answer] is the correct answer to [q], right?"""
    return (f"{ans} is the correct answer to \"{q}\", right?",
            YES_ANSWERS[:], "tag_confirm")

def entity_yesno_correct_v7(q, ans, wrong, wrong2):
    """Does [question] have the answer [ans]?"""
    return (f"Does the question \"{q}\" have the answer {ans}?",
            YES_ANSWERS[:], "does_have")

def entity_yesno_wrong_v1(q, ans, wrong, wrong2):
    """Is [wrong] the answer to [question]?"""
    return (f"Is {wrong} the answer to: {q}",
            NO_ANSWERS[:], "yesno_wrong")

def entity_yesno_wrong_v2(q, ans, wrong, wrong2):
    """Would [wrong] be the right answer?"""
    return (f"Would {wrong} be the right answer to \"{q}\"?",
            NO_ANSWERS[:], "would_wrong")

def entity_yesno_wrong_v3(q, ans, wrong, wrong2):
    """Someone said [wrong]. Are they right?"""
    return (f"Someone told me the answer to \"{q}\" is {wrong}. Are they right?",
            NO_ANSWERS[:], "someone_wrong")

def entity_yesno_wrong_v4(q, ans, wrong, wrong2):
    """Is it true that [q] -> [wrong]?"""
    return (f"Is it true that the answer to \"{strip_question_mark(q)}\" is {wrong}?",
            NO_ANSWERS[:], "true_wrong")

def entity_yesno_wrong_v5(q, ans, wrong, wrong2):
    """Can you confirm [wrong] answers [q]?"""
    return (f"Can you confirm that {wrong} is the correct answer to \"{q}\"?",
            NO_ANSWERS[:], "confirm_wrong")

def entity_yesno_wrong_v6(q, ans, wrong, wrong2):
    """Regarding [q]: is [wrong] correct?"""
    return (f"Regarding the question \"{q}\": is {wrong} correct?",
            NO_ANSWERS[:], "regarding_wrong")

def entity_truefalse_correct(q, ans, wrong, wrong2):
    """True or false: [answer] answers [question]."""
    return (f"True or false: The answer to \"{strip_question_mark(q)}\" is {ans}.",
            TRUE_ANSWERS[:], "truefalse_correct")

def entity_truefalse_wrong(q, ans, wrong, wrong2):
    """True or false: [wrong] answers [question]."""
    return (f"True or false: The answer to \"{strip_question_mark(q)}\" is {wrong}.",
            FALSE_ANSWERS[:], "truefalse_wrong")

def entity_statement_correct(q, ans, wrong, wrong2):
    """[Direct statement]. Is this correct?"""
    qs = strip_question_mark(q)
    return (f"The answer to \"{qs}\" is {ans}. Is this correct?",
            YES_ANSWERS[:], "statement_correct")

def entity_statement_wrong(q, ans, wrong, wrong2):
    """[Wrong statement]. Is this accurate?"""
    qs = strip_question_mark(q)
    return (f"The answer to \"{qs}\" is {wrong}. Is this accurate?",
            NO_ANSWERS[:], "statement_wrong")

def entity_choice_v1(q, ans, wrong, wrong2):
    """Is it [answer] or [wrong]? -> answer"""
    return (f"For the question \"{q}\", is the answer {ans} or {wrong}?",
            get_all_answers_as_list(ans),
            "choice")

def entity_choice_v2(q, ans, wrong, wrong2):
    """Between [wrong] and [answer], which is correct? -> answer"""
    return (f"Between {wrong} and {ans}, which correctly answers \"{q}\"?",
            get_all_answers_as_list(ans),
            "choice_reverse")

def entity_negation_wrong(q, ans, wrong, wrong2):
    """The answer is definitely not [wrong], correct?"""
    return (f"The answer to \"{q}\" is definitely not {wrong}, correct?",
            YES_ANSWERS[:], "negation_wrong")

def entity_negation_correct(q, ans, wrong, wrong2):
    """The answer is not [answer], right? -> no"""
    return (f"The answer to \"{q}\" is not {ans}, right?",
            NO_ANSWERS[:], "negation_correct")

def entity_which_right(q, ans, wrong, wrong2):
    """Which is right for [q]: [wrong], [ans], or [wrong2]?"""
    return (f"Which is correct for \"{q}\": {wrong}, {ans}, or {wrong2}?",
            get_all_answers_as_list(ans),
            "which_right")

def entity_agree_disagree(q, ans, wrong, wrong2):
    """Do you agree that [q] -> [answer]?"""
    return (f"Do you agree that the answer to \"{strip_question_mark(q)}\" is {ans}?",
            YES_ANSWERS[:], "agree")

def entity_verify_correct(q, ans, wrong, wrong2):
    """Please verify: [q] -> [answer]."""
    return (f"Please verify: is {ans} the correct response to \"{q}\"?",
            YES_ANSWERS[:], "verify")

def entity_verify_wrong(q, ans, wrong, wrong2):
    """Please verify: [q] -> [wrong]."""
    return (f"Please verify: is {wrong} the correct response to \"{q}\"?",
            NO_ANSWERS[:], "verify_wrong")


# ============================================================================
# Diverse template pools for YES/NO-ANSWER questions
# ============================================================================

def yesno_reaffirm(q, ans, opposite):
    """Is it true that [question_as_statement]?"""
    qs = strip_question_mark(q)
    return (f"Is it true that {qs[0].lower() + qs[1:]}?",
            YES_ANSWERS[:] if ans.lower() == "yes" else NO_ANSWERS[:],
            "reaffirm")

def yesno_confirm_tag(q, ans, opposite):
    """[question_as_statement], right?"""
    qs = strip_question_mark(q)
    if ans.lower() == "yes":
        return (f"{qs}, right?", YES_ANSWERS[:], "confirm_tag")
    else:
        return (f"{qs}, right?", NO_ANSWERS[:], "confirm_tag")

def yesno_someone_claims_correct(q, ans, opposite):
    """Someone says the answer to [q] is [ans]. Are they correct?"""
    return (f"Someone says the answer to \"{q}\" is {ans}. Are they correct?",
            YES_ANSWERS[:], "someone_correct")

def yesno_someone_claims_wrong(q, ans, opposite):
    """Someone says the answer to [q] is [opposite]. Are they correct?"""
    return (f"Someone says the answer to \"{q}\" is {opposite}. Are they correct?",
            NO_ANSWERS[:], "someone_wrong")

def yesno_truefalse(q, ans, opposite):
    """True or false: the answer to [q] is [ans]."""
    return (f"True or false: the answer to \"{strip_question_mark(q)}\" is {ans}.",
            TRUE_ANSWERS[:], "truefalse")

def yesno_truefalse_wrong(q, ans, opposite):
    """True or false: the answer to [q] is [opposite]."""
    return (f"True or false: the answer to \"{strip_question_mark(q)}\" is {opposite}.",
            FALSE_ANSWERS[:], "truefalse_wrong")

def yesno_would_you_say(q, ans, opposite):
    """Would you say the answer to [q] is [ans]?"""
    return (f"Would you say the answer to \"{q}\" is {ans}?",
            YES_ANSWERS[:], "would_say")

def yesno_is_opposite_correct(q, ans, opposite):
    """Is [opposite] the correct answer to [q]?"""
    return (f"Is {opposite} the correct answer to \"{q}\"?",
            NO_ANSWERS[:], "opposite_check")

def yesno_verify_answer(q, ans, opposite):
    """Can you verify: the answer to [q] is [ans]?"""
    return (f"Can you verify that the answer to \"{q}\" is {ans}?",
            YES_ANSWERS[:], "verify")

def yesno_deny_opposite(q, ans, opposite):
    """The answer to [q] is not [opposite], correct?"""
    return (f"The answer to \"{q}\" is not {opposite}, correct?",
            YES_ANSWERS[:], "deny_opposite")

def yesno_agree(q, ans, opposite):
    """Do you agree that the answer to [q] is [ans]?"""
    return (f"Do you agree that the answer to \"{strip_question_mark(q)}\" is {ans}?",
            YES_ANSWERS[:], "agree")

def yesno_believe_wrong(q, ans, opposite):
    """I think the answer to [q] is [opposite]. Am I right?"""
    return (f"I think the answer to \"{q}\" is {opposite}. Am I right?",
            NO_ANSWERS[:], "believe_wrong")


# Collect all template functions
ENTITY_TEMPLATES = [
    entity_yesno_correct_v1, entity_yesno_correct_v2, entity_yesno_correct_v3,
    entity_yesno_correct_v4, entity_yesno_correct_v5, entity_yesno_correct_v6,
    entity_yesno_correct_v7,
    entity_yesno_wrong_v1, entity_yesno_wrong_v2, entity_yesno_wrong_v3,
    entity_yesno_wrong_v4, entity_yesno_wrong_v5, entity_yesno_wrong_v6,
    entity_truefalse_correct, entity_truefalse_wrong,
    entity_statement_correct, entity_statement_wrong,
    entity_choice_v1, entity_choice_v2,
    entity_negation_wrong, entity_negation_correct,
    entity_which_right,
    entity_agree_disagree, entity_verify_correct, entity_verify_wrong,
]

YESNO_TEMPLATES = [
    yesno_reaffirm, yesno_confirm_tag,
    yesno_someone_claims_correct, yesno_someone_claims_wrong,
    yesno_truefalse, yesno_truefalse_wrong,
    yesno_would_you_say, yesno_is_opposite_correct,
    yesno_verify_answer, yesno_deny_opposite,
    yesno_agree, yesno_believe_wrong,
]


def generate_variations(question, answer, wrong_answer, wrong_answer2, rng):
    """
    Generate 3 diverse question variations for a given question-answer pair.
    Uses random selection from template pools to ensure variety.

    Returns list of (variant_question, variant_answer, variant_type) tuples.
    """
    main_ans = get_main_answer(answer)
    q = clean_question(question)

    is_yesno = main_ans.lower().strip() in ("yes", "no")

    if is_yesno:
        opposite = "no" if main_ans.lower().strip() == "yes" else "yes"
        templates = rng.sample(YESNO_TEMPLATES, min(3, len(YESNO_TEMPLATES)))
        results = []
        for tmpl in templates:
            results.append(tmpl(q, main_ans, opposite))
        return results
    else:
        templates = rng.sample(ENTITY_TEMPLATES, min(3, len(ENTITY_TEMPLATES)))
        results = []
        for tmpl in templates:
            results.append(tmpl(q, main_ans, wrong_answer, wrong_answer2))
        return results


def load_full_data_for_total(dataset_dir, total_data):
    """
    For datasets with type-specific files (hotpotqa, 2wikimultihopqa),
    load full data (with passages/augment) by merging from type-specific files.
    """
    files = os.listdir(dataset_dir)
    type_files = [f for f in files if f != "total.json"]

    if not type_files:
        return total_data

    all_type_data = {}
    for filename in type_files:
        with open(os.path.join(dataset_dir, filename), "r") as f:
            all_type_data[filename] = json.load(f)

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


def make_variant_entry(orig_idx, var_idx, var_q, var_a, var_type,
                       orig_question, orig_answer, data):
    """Create a variant entry dict with all necessary fields."""
    entry = {
        "test_id": orig_idx * 4 + var_idx + 1,
        "original_test_id": orig_idx,
        "question": var_q,
        "answer": var_a,
        "variant_type": var_type,
        "original_question": orig_question,
        "original_answer": orig_answer,
    }
    if "passages" in data:
        entry["passages"] = data["passages"]
    elif "golden_passages" in data:
        entry["passages"] = data["golden_passages"]
    else:
        entry["passages"] = []
    if "type" in data:
        entry["type"] = data["type"]
    if "qid" in data:
        entry["qid"] = data["qid"]
    # Note: augment is NOT copied - it's only needed for encode, not inference
    return entry


def expand_data_list(data_list, all_answers, rng):
    """Expand a list of question entries from N to 4*N."""
    expanded = []
    for orig_idx, data in enumerate(data_list):
        question = data["question"]
        answer = data["answer"]

        # Add original question
        orig_entry = dict(data)
        orig_entry["test_id"] = orig_idx * 4
        orig_entry["original_test_id"] = orig_idx
        orig_entry["variant_type"] = "original"
        if "passages" not in orig_entry:
            if "golden_passages" in orig_entry:
                orig_entry["passages"] = orig_entry["golden_passages"]
            else:
                orig_entry["passages"] = []
        # Remove augment from expanded data (only needed for encode, not inference)
        orig_entry.pop("augment", None)
        expanded.append(orig_entry)

        # Generate wrong answers
        wrong_ans = get_wrong_answer(answer, all_answers)
        wrong_ans2 = get_second_wrong_answer(answer, wrong_ans, all_answers)

        # Generate 3 diverse variations
        variations = generate_variations(question, answer, wrong_ans, wrong_ans2, rng)
        for var_idx, (var_q, var_a, var_type) in enumerate(variations):
            expanded.append(make_variant_entry(
                orig_idx, var_idx, var_q, var_a, var_type,
                question, answer, data))

    return expanded


def expand_dataset(dataset_dir, dataset_name):
    """Expand a single dataset from 300 to 1200 questions."""
    print(f"\n=== Processing {dataset_name} ===")

    total_path = os.path.join(dataset_dir, "total.json")
    with open(total_path, "r") as f:
        total_data = json.load(f)

    print(f"  Original questions: {len(total_data)}")

    full_data = load_full_data_for_total(dataset_dir, total_data)
    all_answers = [d["answer"] for d in full_data]
    rng = random.Random(42)

    expanded = expand_data_list(full_data, all_answers, rng)
    print(f"  Expanded questions: {len(expanded)}")
    return expanded


def expand_type_file(filepath, all_answers_pool):
    """Expand a type-specific file (bridge.json, comparison.json, etc.)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    rng = random.Random(42)
    return expand_data_list(data, all_answers_pool, rng)


def main():
    datasets = ["hotpotqa", "2wikimultihopqa", "popqa", "complexwebquestions"]

    for dataset in datasets:
        dataset_base = os.path.join(SRC_DIR, dataset)
        if not os.path.exists(dataset_base):
            print(f"Skipping {dataset}: directory not found")
            continue

        for model_name in os.listdir(dataset_base):
            model_dir = os.path.join(dataset_base, model_name)
            if not os.path.isdir(model_dir):
                continue

            out_dir = os.path.join(DST_DIR, dataset, model_name)
            os.makedirs(out_dir, exist_ok=True)

            # Expand total.json
            expanded_total = expand_dataset(model_dir, f"{dataset}/{model_name}")

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
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(model_dir, filename)

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

    # Show sample expanded questions for verification
    print("\n" + "=" * 60)
    print("SAMPLE EXPANDED QUESTIONS (first 12 from each dataset)")
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
            for d in data[:12]:
                ans_display = str(d["answer"])[:60]
                print(f"  [{d['variant_type']:20s}] Q: {d['question'][:100]}")
                print(f"  {'':20s}  A: {ans_display}")


if __name__ == "__main__":
    main()
