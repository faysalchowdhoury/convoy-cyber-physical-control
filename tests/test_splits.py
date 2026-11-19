"""Anti-leakage: splits must be vehicle-disjoint."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.platoon_graph import make_splits


def _toy_strings():
    # 30 strings, members drawn from disjoint vehicle blocks, varied start frames
    strings = []
    vid = 0
    for k in range(30):
        members = [vid, vid + 1, vid + 2, vid + 3]
        vid += 4
        strings.append({"members": members, "f_start": k * 50})
    return strings


def test_vehicle_disjoint_splits():
    strings = _toy_strings()
    splits = make_splits(strings, {"train": 0.6, "val": 0.2, "test": 0.2})
    sets = {}
    for name, idxs in splits.items():
        v = set()
        for i in idxs:
            v.update(strings[i]["members"])
        sets[name] = v
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


def test_all_strings_assigned_or_dropped_consistently():
    strings = _toy_strings()
    splits = make_splits(strings, {"train": 0.6, "val": 0.2, "test": 0.2})
    total = sum(len(v) for v in splits.values())
    # disjoint-block strings should all be assignable
    assert total == len(strings)


def test_chronological_ordering_respected():
    strings = _toy_strings()
    splits = make_splits(strings, {"train": 0.6, "val": 0.2, "test": 0.2})
    max_train_start = max(strings[i]["f_start"] for i in splits["train"])
    min_test_start = min(strings[i]["f_start"] for i in splits["test"])
    assert max_train_start <= min_test_start
