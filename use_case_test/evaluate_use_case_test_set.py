"""Evaluate the human-reviewed use-case set with the final tuned BERT model.

This script performs inference only. It refuses to evaluate provisional labels.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Ensure model/tokenizer loading cannot silently reach Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = SCRIPT_DIR / "use_case_test_set_reviewed.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "Models" / "best_hp_config_maxlen_low"
PREDICTIONS_PATH = SCRIPT_DIR / "use_case_test_predictions.csv"
METRICS_PATH = SCRIPT_DIR / "use_case_test_metrics.json"
CONFUSION_MATRIX_PATH = SCRIPT_DIR / "use_case_test_confusion_matrix.png"
INDEPENDENT_CONFUSION_MATRIX_PATH = (
    SCRIPT_DIR / "use_case_test_confusion_matrix_independent.png"
)
DEPENDENT_CONFUSION_MATRIX_PATH = (
    SCRIPT_DIR / "use_case_test_confusion_matrix_dependent.png"
)

MAX_LENGTH = 64
BATCH_SIZE = 32
EXPECTED_ROWS = 100
LABEL_NAMES = ["not_sarcastic", "sarcastic"]


def clean(text: object) -> str:
    """Apply the same text cleaning used in the existing notebook."""
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    cleaned = re.sub(r"<.*?>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def validate_reviewed_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "id",
        "text",
        "generated_label",
        "generated_label_name",
        "use_case_category",
        "review_status",
        "final_label",
        "reviewer_notes",
        "context",
        "context_dependency",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}.")
    if frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError("Every row must have a unique, non-empty id.")
    if frame["text"].isna().any():
        raise ValueError("Text cannot be empty.")

    pending = frame["review_status"].astype(str).str.strip().str.lower() != "reviewed"
    if pending.any():
        ids = frame.loc[pending, "id"].tolist()
        raise ValueError(
            "Human review is incomplete. Set review_status='reviewed' and provide "
            f"final_label for every row. Pending row ids: {ids}"
        )

    numeric_labels = pd.to_numeric(frame["final_label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        bad_ids = frame.loc[numeric_labels.isna() | ~numeric_labels.isin([0, 1]), "id"].tolist()
        raise ValueError(f"final_label must be 0 or 1 for every row. Invalid row ids: {bad_ids}")

    reviewed = frame.copy()
    reviewed["final_label"] = numeric_labels.astype(int)
    reviewed["context_dependency"] = (
        reviewed["context_dependency"].astype(str).str.strip().str.lower()
    )
    invalid_dependency = ~reviewed["context_dependency"].isin(["independent", "dependent"])
    if invalid_dependency.any():
        raise ValueError(
            "context_dependency must be 'independent' or 'dependent'. Invalid row ids: "
            f"{reviewed.loc[invalid_dependency, 'id'].tolist()}"
        )

    reviewed["cleaned_text"] = reviewed["text"].map(clean)
    empty = reviewed["cleaned_text"].str.len() == 0
    if empty.any():
        raise ValueError(f"Text is empty after preprocessing for row ids: {reviewed.loc[empty, 'id'].tolist()}")
    return reviewed


def load_final_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    required_files = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = [name for name in sorted(required_files) if not (CHECKPOINT_PATH / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Final checkpoint is incomplete at {CHECKPOINT_PATH}; missing: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH, local_files_only=True
    )
    if model.config.model_type != "bert":
        raise ValueError(
            f"Expected the final tuned BERT checkpoint, found model_type={model.config.model_type!r}."
        )
    if model.config.num_labels != 2:
        raise ValueError(f"Expected a binary classifier, found num_labels={model.config.num_labels}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    predictions: list[int] = []
    sarcasm_probabilities: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions.extend(probabilities.argmax(dim=-1).cpu().tolist())
            sarcasm_probabilities.extend(probabilities[:, 1].cpu().tolist())

    return predictions, sarcasm_probabilities


def calculate_metrics(gold: list[int], predictions: list[int]) -> dict:
    matrix = confusion_matrix(gold, predictions, labels=[0, 1])
    return {
        "n": len(gold),
        "label_counts": {
            "not_sarcastic": int(sum(label == 0 for label in gold)),
            "sarcastic": int(sum(label == 1 for label in gold)),
        },
        "accuracy": float(accuracy_score(gold, predictions)),
        "precision": float(precision_score(gold, predictions, zero_division=0)),
        "recall": float(recall_score(gold, predictions, zero_division=0)),
        "f1": float(f1_score(gold, predictions, zero_division=0)),
        "f1_macro": float(f1_score(gold, predictions, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            gold,
            predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    gold: list[int], predictions: list[int], title: str, output_path: Path
) -> None:
    matrix = confusion_matrix(gold, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=LABEL_NAMES).plot(
        ax=axis, cmap="Blues", colorbar=False
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Use-case dataset not found: {DATA_PATH}")

    reviewed = validate_reviewed_data(pd.read_csv(DATA_PATH, keep_default_na=False))
    tokenizer, model, device = load_final_model()
    predictions, probabilities = predict(
        # Context is intentionally excluded: BERT receives only cleaned target text.
        reviewed["cleaned_text"].tolist(), tokenizer, model, device
    )

    output = reviewed.copy()
    output["model_prediction"] = predictions
    output["predicted_label_name"] = [LABEL_NAMES[label] for label in predictions]
    output["prob_not_sarcastic"] = [1.0 - probability for probability in probabilities]
    output["prob_sarcastic"] = probabilities
    output["confidence"] = [
        probability if prediction == 1 else 1.0 - probability
        for prediction, probability in zip(predictions, probabilities)
    ]
    output["correct"] = (output["model_prediction"] == output["final_label"]).astype(int)
    prediction_columns = [
        "id",
        "text",
        "context",
        "final_label",
        "model_prediction",
        "predicted_label_name",
        "prob_not_sarcastic",
        "prob_sarcastic",
        "confidence",
        "correct",
        "context_dependency",
    ]
    output[prediction_columns].to_csv(PREDICTIONS_PATH, index=False)

    gold = output["final_label"].tolist()
    overall = calculate_metrics(gold, predictions)
    by_context_dependency = {}
    context_matrix_paths = {
        "independent": INDEPENDENT_CONFUSION_MATRIX_PATH,
        "dependent": DEPENDENT_CONFUSION_MATRIX_PATH,
    }
    for dependency in ["independent", "dependent"]:
        mask = output["context_dependency"] == dependency
        subset_gold = output.loc[mask, "final_label"].tolist()
        subset_predictions = output.loc[mask, "model_prediction"].tolist()
        by_context_dependency[dependency] = calculate_metrics(
            subset_gold, subset_predictions
        )
        save_confusion_matrix(
            subset_gold,
            subset_predictions,
            f"Final tuned BERT — {dependency} examples",
            context_matrix_paths[dependency],
        )

    metrics = {
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "model_type": model.config.model_type,
        "max_length": MAX_LENGTH,
        "model_input_column": "text",
        "context_used_as_model_input": False,
        "ground_truth_column": "final_label",
        "overall": overall,
        "by_context_dependency": by_context_dependency,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    save_confusion_matrix(
        gold,
        predictions,
        "Final tuned BERT — reviewed use-case test set",
        CONFUSION_MATRIX_PATH,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved independent confusion matrix: {INDEPENDENT_CONFUSION_MATRIX_PATH}")
    print(f"Saved dependent confusion matrix: {DEPENDENT_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
