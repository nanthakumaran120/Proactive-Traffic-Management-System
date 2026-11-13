"""Train and chronologically evaluate Gandhipuram traffic forecasting models.

Uses ONLY datasets/gandhipuram_coimbatore_traffic_synthetic.csv and saves
trained classification + regression pipelines and evaluation metadata under models/.
"""

from __future__ import annotations

import sys

from src.future_predictor import MODELS_DIR, train_and_evaluate_models


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    metadata = train_and_evaluate_models()
    val = metadata["validation_report"]

    print("=" * 65)
    print("  GANDHIPURAM FUTURE TRAFFIC FORECASTING - TRAINING REPORT")
    print("=" * 65)
    print(f"Dataset           : {val['dataset_path']}")
    print(f"Raw rows          : {val['raw_row_count']} (removed {val['empty_rows_removed']} empty rows)")
    print(f"Valid rows        : {val['valid_row_count']} across {val['junction_count']} junctions")
    print(f"Date coverage     : {val['timestamp_start']} -> {val['timestamp_end']}")
    print(f"Forecast pairs    : {metadata['total_horizon_pairs']}")
    print(f"Chrono eval dates : {', '.join(metadata['chronological_eval_dates'])}")

    print("\n--- Classification Chronological Expanding-Window Results ---")
    for row in metadata["classifier_comparison"]:
        print(
            f"  {row['model']:22s} | Acc={row['accuracy']:.4f} | "
            f"Prec={row['precision']:.4f} | Rec={row['recall']:.4f} | F1={row['f1_score']:.4f}"
        )
    pers = metadata["persistence_baseline"]
    print(f"  {pers['model']:42s} | Acc={pers['accuracy']:.4f} | F1={pers['f1_score']:.4f}")
    print(f"Selected Classifier: {metadata['best_classifier']}")

    print("\n--- Regression Chronological Expanding-Window Results ---")
    for tgt, rows in metadata["regression_comparison"].items():
        print(f"  Target: {tgt} (Selected: {metadata['best_regressors'][tgt]})")
        for r in rows:
            print(f"    {r['model']:25s} | MAE={r['mae']:.3f} | RMSE={r['rmse']:.3f}")

    print(f"\nSaved artifacts directory: {MODELS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
