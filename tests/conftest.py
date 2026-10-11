"""Shared synthetic-data builders for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_labels():
    """Factory for a synthetic labels table shaped like labels.parquet.

    Sites and cohorts are laid out so that every institution x cohort x event
    stratum is populated; times are positive and events roughly balanced.
    """

    def _build(
        site_sizes: dict[str, int] | None = None,
        *,
        endpoint: str = "pfs",
        event_rate: float = 0.5,
        seed: int = 0,
    ) -> pd.DataFrame:
        site_sizes = site_sizes or {"TUM": 40, "LMU": 30, "Erlangen": 30, "Wuerzburg": 10}
        rng = np.random.default_rng(seed)
        rows = []
        counter = 0
        for site, size in site_sizes.items():
            for i in range(size):
                counter += 1
                cohort = "definitive" if i % 2 == 0 else "neoadjuvant"
                rows.append(
                    {
                        "patient_key": f"pt_{counter:03d}",
                        "cohort": cohort,
                        "institution": site,
                        f"{endpoint}_event": int(rng.random() < event_rate),
                        f"{endpoint}_time": float(rng.integers(30, 2000)),
                    }
                )
        return pd.DataFrame(rows)

    return _build
