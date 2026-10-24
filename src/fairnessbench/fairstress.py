"""
FairStress: Asymmetrically Unfair Decision Making and Reasoning in LLMs

Allen Institute for AI (Ai2)
Dataset: https://huggingface.co/datasets/allenai/fairstress
Code:    https://github.com/pardissz/fairstress

FairStress poses two-candidate decisions (hiring, healthcare, lending, housing,
...) where the candidates' demographic identity is signalled at one of four
degrees. Each scenario has a ground-truth version (one candidate is objectively
stronger) and an ambiguous version (a true tie), optionally preceded by an
adversarial prefix that argues for one candidate. Fairness is measured as a
decision process: is the model right, firm and even-handed equally often for
minority- and majority-coded candidates?

# eval the IRT-selected FairStress-Core split (recommended)
inspect eval src/fairnessbench/fairstress.py@fairstress

# only ground-truth items, explicit identity labels (signaling level 3)
inspect eval src/fairnessbench/fairstress.py@fairstress -T condition=GT -T signaling_levels=3

# only items without an adversarial prefix
inspect eval src/fairnessbench/fairstress.py@fairstress -T include_injections=false
"""

import json
from collections.abc import Callable, Sequence
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from fairnessbench.scorer import fairstress_scorer

DATASET_PATH = "allenai/fairstress"
# Pinned to a specific `main` branch commit so results stay reproducible.
FAIRSTRESS_DATASET_REVISION = "a5cdf125f8a59d605efd0ea9deefe75083588208"

# The dataset card's YAML points the `core` split at a folder that does not
# exist (the shard actually lives in `data/`), so the parquet shards are
# addressed explicitly. Both globs are relative to the dataset repo root.
DATA_FILES: dict[str, str] = {
    "core": "data/core-*.parquet",
    "full": "data/full-*.parquet",
}

Split = Literal["core", "full"]
Condition = Literal["GT", "AMB", "all"]


@task
def fairstress(
    split: Split = "core",
    condition: Condition = "all",
    signaling_levels: int | Sequence[int] | None = None,
    include_injections: bool = True,
    shuffle: bool = True,
    max_tokens: int | None = None,
) -> Task:
    """Inspect task implementing the FairStress fairness benchmark.

    Args:
        split: ``"core"`` (default) is the ~50K-item FairStress-Core subset
            selected with 2PL IRT; ``"full"`` is the complete ~13M-item corpus
            used in the paper (multi-GB download; pair with ``--limit``).
        condition: ``"GT"`` keeps only ground-truth items (one candidate is
            objectively stronger), ``"AMB"`` only ambiguous tie items, and
            ``"all"`` (default) keeps both. Metrics that need the excluded
            condition are reported as NaN.
        signaling_levels: Restrict to identity-signalling degrees 0 (none)
            to 3 (explicit label). ``None`` (default) keeps all levels.
        include_injections: Whether to keep items that carry an adversarial
            prefix (default True). The Robustness Gap and Tie-Break Shift
            metrics require them.
        shuffle: Whether to shuffle the dataset order (default True).
        max_tokens: Optional cap on generated tokens. The paper used 32-64
            for non-reasoning models; leave unset for reasoning models.
    """
    if split not in DATA_FILES:
        raise ValueError(f"split must be one of {sorted(DATA_FILES)}, got {split!r}")
    if condition not in ("GT", "AMB", "all"):
        raise ValueError(f"condition must be 'GT', 'AMB' or 'all', got {condition!r}")

    dataset = hf_dataset(
        path=DATASET_PATH,
        split=split,
        data_files={split: DATA_FILES[split]},
        sample_fields=record_to_sample(
            condition=condition,
            signaling_levels=signaling_levels,
            include_injections=include_injections,
        ),
        shuffle=shuffle,
        revision=FAIRSTRESS_DATASET_REVISION,
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=fairstress_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=max_tokens),
    )


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Normalise a nested field that may be a dict, JSON string or null."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip() or value.strip() == "null":
            return None
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def _nested(record: dict[str, Any], key: str, field: str) -> Any:
    """Read ``record[key][field]``, tolerating flattened parquet columns."""
    obj = _as_dict(record.get(key))
    if obj is not None:
        return obj.get(field)
    for flat in (f"{key}.{field}", f"{key}_{field}"):
        if flat in record:
            return record[flat]
    return None


def _normalise_levels(levels: int | Sequence[int] | None) -> set[int] | None:
    if levels is None:
        return None
    if isinstance(levels, int):
        return {levels}
    return {int(lvl) for lvl in levels}


def record_to_sample(
    condition: Condition = "all",
    signaling_levels: int | Sequence[int] | None = None,
    include_injections: bool = True,
) -> Callable[[dict[str, Any]], list[Sample]]:
    """Build a record-to-sample converter that also applies the task filters.

    Returns a closure matching the ``sample_fields`` callable signature of
    ``hf_dataset``. Records excluded by a filter map to an empty list, so
    filtering happens while the dataset is read rather than after millions of
    ``Sample`` objects have been built.
    """
    levels = _normalise_levels(signaling_levels)

    def converter(record: dict[str, Any]) -> list[Sample]:
        cond = str(record["condition"])
        level = int(record["signaling_level"])
        injection = _as_dict(record.get("injection"))
        has_injection = bool(record.get("has_injection", injection is not None))

        if condition != "all" and cond != condition:
            return []
        if levels is not None and level not in levels:
            return []
        if not include_injections and has_injection:
            return []

        expected = record.get("expected_correct") or None
        # Target is the slot letter of the stronger candidate, or empty for
        # ambiguous (tie) items where no answer is objectively correct.
        target = str(expected)[-1] if expected else ""

        if injection is None and has_injection:
            injection = {
                "id": _nested(record, "injection", "id"),
                "family": _nested(record, "injection", "family"),
                "class": _nested(record, "injection", "class"),
                "direction": _nested(record, "injection", "direction"),
            }
        injection = injection or {}

        metadata: dict[str, Any] = {
            "scenario_id": record.get("scenario_id"),
            "domain": record.get("domain"),
            "condition": cond,
            "signaling_level": level,
            "contrast_id": record.get("contrast_id"),
            "contrast_category": record.get("contrast_category"),
            "minority_group": record.get("minority_group"),
            "majority_group": record.get("majority_group"),
            "group_A": _nested(record, "candidate_A", "group"),
            "group_B": _nested(record, "candidate_B", "group"),
            "minority_slot": record.get("minority_slot"),
            "expected_correct": expected,
            "has_injection": has_injection,
            "injection_id": injection.get("id"),
            "injection_family": injection.get("family") or "none",
            "injection_class": injection.get("class"),
            "injection_resolved": record.get("injection_resolved") or "neutral",
            "injection_targets": record.get("injection_targets") or "neither",
        }

        return [
            Sample(
                id=record["item_id"],
                input=record["prompt"],
                target=target,
                metadata=metadata,
            )
        ]

    return converter
