# NLP Final Project: Sarcasm Detection

This repository contains an academic binary sarcasm-detection project using transformer-based language models.

## Project structure

- `binary_model_NLP.ipynb` — main Colab notebook for preprocessing, model comparison, BERT hyperparameter tuning, final validation/test evaluation, and the interactive Colab prototype.
- `use_case_test/use_case_test_set_reviewed.csv` — final human-reviewed use-case test dataset containing short social-media, chatbot, and customer-service comments.
- `use_case_test/evaluate_use_case_test_set.py` — inference-only script used to evaluate the final hyperparameter-tuned BERT model on the use-case dataset.
- `use_case_test/use_case_test_predictions.csv` — per-example model predictions, probabilities, confidence scores, and correctness indicators.
- `use_case_test/use_case_test_metrics.json` — overall evaluation metrics and separate results for context-independent and context-dependent examples.
- `use_case_test/use_case_test_confusion_matrix*.png` — overall and context-group confusion-matrix visualizations.

The use-case evaluation was added as a separate final-project evaluation step. It is therefore implemented in a standalone Python script rather than inside the original training notebook. Conversational context is retained for human labeling and error analysis, while the model receives only the target comment text.
