"""Discrimination metrics shared by the model fits and the paper scripts.

- :func:`harrell_c`: Harrell's C-index (selection in the inner CV; Table 2).
- :func:`ipcw_auc_pooled`: Uno's IPCW cumulative/dynamic time-dependent AUC at
  one horizon (scikit-survival, imported lazily).
- :func:`rank_normal`: rank-based normal scores, used to put the risks of the
  five fold models of one repeat on a common scale before pooling them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

DAYS_PER_MONTH = 365.25 / 12


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> tuple[float, int]:
    """Harrell's C-index for right-censored data — returns ``(c, n_pairs)``.

    A pair is comparable when the one that failed did so strictly before the
    other's observed time; it is concordant when the higher risk score belongs
    to the earlier failure. Tied risk scores count a half. Equal observed times
    are not comparable (the strict convention).

    Kept as an explicit implementation (rather than delegating to lifelines)
    because the comparable-pair count is needed to skip degenerate bootstrap
    draws; a unit test asserts exact agreement with
    ``lifelines.utils.concordance_index``.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event).astype(int)
    r = np.asarray(risk, dtype=float)
    num = den = 0.0
    for i in np.flatnonzero(e == 1):
        later = t > t[i]
        n_later = int(later.sum())
        if not n_later:
            continue
        den += n_later
        num += float((r[i] > r[later]).sum()) + 0.5 * float((r[i] == r[later]).sum())
    return (num / den if den else float("nan")), int(den)


def rank_normal(x: np.ndarray) -> np.ndarray:
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def ipcw_auc_pooled(frame_truth, score, days):
    """Uno's IPCW cumulative/dynamic AUC at one horizon (days); NaN when undefined.

    The censoring distribution is estimated on the same patients that are scored.
    """
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    y = Surv.from_arrays(
        frame_truth["event"].astype(bool).to_numpy(), frame_truth["time"].to_numpy()
    )
    try:
        auc, _ = cumulative_dynamic_auc(y, y, np.asarray(score), np.array([days]))
        return float(auc[0])
    except ValueError:
        return float("nan")
