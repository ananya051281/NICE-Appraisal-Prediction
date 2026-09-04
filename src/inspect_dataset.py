"""Terminal report for the small appraisal-level pilot dataset."""
from __future__ import annotations

from .data_loader import BOOLEAN_COLUMNS, load_ml_data
from .features import FEATURE_COLUMNS


def dataset_report() -> str:
    data = load_ml_data()
    counts = data.decision_label.value_counts().to_dict()
    lines = [
        "NICE ML appraisal pilot dataset inspection",
        "=" * 44,
        f"Total ML records: {len(data)}",
        f"Favourable (1): {counts.get(1, 0)}",
        f"Unfavourable (0): {counts.get(0, 0)}",
        f"No decision (2): {counts.get(2, 0)}",
        f"Usable binary records: {int(data.decision_label.isin([0, 1]).sum())}",
        f"Label distribution: {counts}",
        f"Independent appraisal groups: {data.group_id.nunique()}",
        f"Guidance types: {data.guidance_type.value_counts(dropna=False).to_dict()}",
        f"Available leakage-safe features: {', '.join(FEATURE_COLUMNS)}",
        "Missing values:",
    ]
    lines.extend(f"  {column}: {int(count)}" for column, count in data.isna().sum().items())
    lines.append("Feature summaries:")
    for column in BOOLEAN_COLUMNS:
        lines.append(f"  {column}: {data[column].value_counts(dropna=False).to_dict()}")
    lines.append(f"  icer_band: {data.icer_band.value_counts(dropna=False).to_dict()}")
    lines.append(f"  publication years: {data.publication_date.dt.year.value_counts(dropna=False).sort_index().to_dict()}")
    lines.extend([
        "WARNING: This pilot has only 9 independent groups and 7 binary records.",
        "WARNING: It is unsuitable for reliable validation, tuning, or generalisation claims.",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print(dataset_report())
