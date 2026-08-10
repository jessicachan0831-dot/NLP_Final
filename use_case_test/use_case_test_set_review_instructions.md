# Human review instructions for the use-case test set

## Purpose

`use_case_test_set.csv` is a separate, generated use-case test set for short social-media, chatbot, and customer-service comments. It was not sampled from or merged with the Kaggle training, validation, or test files.

The `generated_label` values are provisional annotations only. They must not be treated as final ground truth or used for evaluation until a human reviewer has approved every row.

Label definitions:

- `0` — not sarcastic
- `1` — sarcastic

## Review procedure

1. Review each comment independently, without looking at model predictions.
2. Decide whether the intended reading is sarcastic or not sarcastic.
3. Enter the human decision as `0` or `1` in `final_label`.
4. Change `review_status` from `pending_human_review` to `approved`.
5. Use `reviewer_notes` to record uncertainty, a changed label, or context that would be needed for a confident decision.
6. Pay special attention to rows whose `use_case_category` is `ambiguous_short`. Short comments often cannot be labeled reliably without conversational context. If a row is too ambiguous, replace it with a clearer comment and review the replacement rather than approving an uncertain label.
7. Confirm that all 100 rows are approved and the final labels remain balanced: exactly 50 sarcastic (`1`) and 50 not sarcastic (`0`). If review changes the balance, replace or revise examples and review them until the final set is 50/50.

Do not add model predictions to the CSV during review. This avoids biasing the human labels.

## Pre-evaluation checks

The evaluation script intentionally stops unless all of the following are true:

- The CSV contains exactly 100 unique rows.
- Every `review_status` is `approved`.
- Every `final_label` is either `0` or `1`.
- The final labels contain exactly 50 examples of each class.
- The tuned checkpoint exists at `Models/best_hp_config_maxlen_low`.

After review is complete, run the evaluator from the project root:

```bash
.venv/bin/python use_case_test/evaluate_use_case_test_set.py
```

It performs inference only. It does not train, fine-tune, or modify the checkpoint.
