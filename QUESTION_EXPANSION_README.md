# Question Expansion: 300 → 1200

## Task 1: How Answer Correctness Is Judged

### Principle

The evaluation logic is in `src/utils.py`, specifically the `evaluate()` function and the `BaseDataset` class methods.

**Step 1: Prediction Extraction**

When the model generates a response, the `evaluate()` function first truncates the raw output at the first occurrence of `.`, `\n`, or `,` to extract just the core answer:

```python
pred = pred.strip()
stop_list = [".", "\n", ","]
for stop in stop_list:
    end_pos = pred.find(stop)
    if end_pos != -1:
        pred = pred[:end_pos].strip()
```

For example, if the model outputs `"politician. He served in both the House..."`, it becomes `"politician"`.

**Step 2: Answer Normalization**

Both the prediction and each ground truth answer are normalized via `normalize_answer()`:
1. Convert to lowercase: `"Politician"` → `"politician"`
2. Remove punctuation: `"polit."` → `"polit"`
3. Remove articles (a, an, the): `"the politician"` → `"politician"`
4. Fix whitespace: collapse multiple spaces into one

**Step 3: Exact Match (EM)**

The normalized prediction is compared against ALL ground truth options. If it matches any one of them, EM = 1 (correct); otherwise EM = 0.

```python
correct = max([int(normalize_answer(prediction) == normalize_answer(gt)) for gt in ground_truths])
```

**Step 4: F1 Score**

Token-level overlap between prediction and ground truth:
- Tokenize both normalized strings by whitespace
- Count common tokens
- Calculate precision = common / prediction_tokens
- Calculate recall = common / ground_truth_tokens
- F1 = 2 * precision * recall / (precision + recall)

The best F1 across all ground truth options is reported.

### Example

**Question**: `"What is George Rankin's occupation?"`
**Ground truth answers**: `["politician", "political leader", "political figure", "polit.", "pol"]`
**Model output**: `"politician. He served in both the House of Representatives"`

1. **Truncation** at first `.`: `"politician"`
2. **Normalization**: `"politician"` → `"politician"` (no change)
3. **EM check** against each ground truth:
   - `normalize("politician") == normalize("politician")` → `"politician" == "politician"` → ✓ **Match!**
   - EM = 1
4. **F1**: tokens `["politician"]` vs `["politician"]` → precision=1.0, recall=1.0, F1=1.0

**Another example with a yes/no variation**:

**Question**: `"Is George Rankin's occupation politician?"`
**Ground truth answers**: `["yes", "Yeah", "correct", "right", "true", "Yes"]`
**Model output**: `"Yes, that is correct."`

1. **Truncation** at first `,`: `"Yes"`
2. **Normalization**: `"Yes"` → `"yes"`
3. **EM check**:
   - `normalize("Yes") == normalize("yes")` → `"yes" == "yes"` → ✓ **Match!**
   - EM = 1

---

## Task 2: Question Expansion

### Approach

Each of the 300 original questions follows the pattern `"What is [NAME]'s occupation?"`. For each, three variations are generated:

| # | Variation Type | Question Template | Expected Answer |
|---|---------------|------------------|-----------------|
| 0 | Original | What is [NAME]'s occupation? | [occupation aliases] |
| 1 | Yes/No (correct) | Is [NAME]'s occupation [correct_occupation]? | yes, Yeah, correct, right, true, Yes |
| 2 | Yes/No (wrong) | Is [NAME]'s occupation [wrong_occupation]? | no, No, incorrect, wrong, false, nope |
| 3 | Statement | [NAME] is a/an [correct_occupation]? | yes, Yeah, correct, right, true, Yes |

### Files

- **Original**: `popqa/qwen2.5-1.5b-instruct/total.json` (300 entries, unchanged)
- **Expanded**: `popqa/qwen2.5-1.5b-instruct/total_expanded.json` (1200 entries)

### New Fields

Two metadata fields are added (do not affect inference):
- `original_test_id`: maps back to the original question's test_id (0-299)
- `variation_type`: one of `"original"`, `"yes_no_correct"`, `"yes_no_wrong"`, `"statement_correct"`

### Answer Design

The yes/no answer lists are comprehensive to account for different model response styles:
- **Yes answers**: `["yes", "Yeah", "correct", "right", "true", "Yes"]`
- **No answers**: `["no", "No", "incorrect", "wrong", "false", "nope"]`

These work with the existing evaluation pipeline because:
1. The `evaluate()` function truncates at `.`, `\n`, `,` — so `"Yes, he is a politician."` becomes `"Yes"`
2. `normalize_answer()` lowercases — so `"Yes"` becomes `"yes"`, matching the ground truth

### Wrong Occupation Selection

For "yes/no wrong" variations, a random occupation is selected from the pool of 65 unique occupations found in the dataset, ensuring it does NOT appear in the question's correct answer list.

### Usage

To use the expanded file for inference with misinfo modes:

```bash
# Replace 'total' with 'total_expanded' in the data_type argument, or
# copy total_expanded.json to a new data directory
python src/inference.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset popqa \
    --data_type total_expanded \
    --inference_method misinfo_plain \
    --max_new_tokens 20 \
    --num_train_epochs 15 \
    --lora_rank 2 \
    --lora_alpha 32
```

Note: The expanded file is fully compatible with `misinfo_plain`, `misinfo_icl`, and `misinfo_prag` inference modes. For `prag` and `combine` modes, only the original entries (every 4th starting from index 0) will have matching LoRA adapters.
