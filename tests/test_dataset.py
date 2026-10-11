"""PHI-boundary tests: the split between the id map and the de-identified table.

`deidentify` is the single point where direct identifiers leave the analysis
path. If it ever passes one through, the de-identified table stops being
de-identified — so the test enumerates columns on both sides.
"""

from __future__ import annotations

import pandas as pd

from ec_wsi_2026.clinical import schema as sch
from ec_wsi_2026.dataset import deidentify


def _coerced_like() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_key": ["def_001", "neo_002"],
            "cohort": ["definitive", "neoadjuvant"],
            "source_row": [2, 3],
            "patient_id": ["MRN-123", "MRN-456"],
            "date_of_birth": pd.to_datetime(["1950-01-01", "1960-06-15"]),
            "age_at_crt": [70.0, 62.0],
            "pfs_event": [1, 0],
        }
    )


class TestDeidentify:
    def test_phi_columns_are_dropped_from_the_analysis_table(self):
        _, deidentified = deidentify(_coerced_like())
        for column in sch.PHI_COLUMNS:
            assert column not in deidentified.columns

    def test_phi_columns_survive_in_the_id_map_with_the_key(self):
        id_map, _ = deidentify(_coerced_like())
        for column in ("patient_key", *sch.PHI_COLUMNS):
            assert column in id_map.columns

    def test_nothing_else_is_dropped(self):
        source = _coerced_like()
        _, deidentified = deidentify(source)
        expected = [c for c in source.columns if c not in sch.PHI_COLUMNS]
        assert list(deidentified.columns) == expected
        assert len(deidentified) == len(source)
