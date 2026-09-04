#!/usr/bin/env python3.12
"""Fake-recall at real-recall floors, from a results_selop.txt (unified format).

For each real-recall floor f, pick the threshold tau that makes real-recall == f
(tau = f-quantile of real fake-scores; a real is predicted real iff score < tau),
then report fake-recall = fraction of fakes with score >= tau at that tau.

  python3.12 fake_recall_at_floors.py runs/test_full/results_selop.txt
"""
import re
import sys

import numpy as np

LINE = re.compile(r"^(OK|XX)\s+truth=(real|fake)\s+pred=\S+\s+type=\S+\s+fake=([0-9.]+)\s+match=\S+\s+(.*)$")


def load(path):
    real, fake = [], []
    with open(path) as fh:
        for ln in fh:
            m = LINE.match(ln.rstrip("\n"))
            if not m:
                continue
            _, truth, score, _ = m.groups()
            (real if truth == "real" else fake).append(float(score))
    return np.array(real), np.array(fake)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/test_full/results_selop.txt"
    floors = [0.80, 0.85, 0.90, 0.95, 0.98]
    real, fake = load(path)
    print(f"file: {path}")
    print(f"n_real={len(real)}  n_fake={len(fake)}")
    print(f"AUC(real-vs-fake)={_auc(real, fake):.4f}\n")
    print(f"{'real_floor':>10} {'threshold':>10} {'real_recall':>12} {'fake_recall':>12}")
    print("-" * 48)
    for f in floors:
        tau = float(np.quantile(real, f))            # f-fraction of reals fall below tau
        rr = float((real < tau).mean())
        fr = float((fake >= tau).mean())
        print(f"{f*100:>9.0f}% {tau:>10.4f} {rr*100:>11.2f}% {fr*100:>11.2f}%")


def _auc(real, fake):
    # rank-based AUC = P(fake_score > real_score)
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    s = np.r_[real, fake]
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {i: (csum[i] - cnt[i] / 2 + 0.5) for i in range(len(cnt))}
    ranks = np.array([avg[i] for i in inv])
    n_pos, n_neg = len(fake), len(real)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


if __name__ == "__main__":
    main()
