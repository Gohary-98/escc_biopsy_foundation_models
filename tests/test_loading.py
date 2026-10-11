"""Institution cross-check tests.

The institution appears twice — as free text and as the patient-number block —
and the redundancy is the safety net: a contradiction means either mojibake or a
patient filed under the wrong number. The check WARNS and records rather than
raising (a single miskeyed row should not block the whole cohort), so the tests
assert on `institution_matches_block` and the ParseLog, not on an exception.
"""

from __future__ import annotations

import pandas as pd

from ec_wsi_2026.clinical import loading
from ec_wsi_2026.clinical.coercion import ParseLog


def _run_cross_check(institution_text: str, cohort: str, patient_no: int):
    out = pd.DataFrame({"cohort": [cohort], "patient_no": pd.array([patient_no], dtype="Int64")})
    raw = pd.DataFrame({"cohort": [cohort], "source_row": [2], "institution": [institution_text]})
    log = ParseLog()
    loading._resolve_institution(out, raw, log)
    return out, log


class TestInstitutionCrossCheck:
    def test_agreement_passes_silently(self):
        # definitive patient 10 sits in the TUM block (1-63).
        out, log = _run_cross_check("TUM", "definitive", 10)
        assert out.loc[0, "institution"] == "TUM"
        assert bool(out.loc[0, "institution_matches_block"])
        assert len(log) == 0

    def test_contradiction_is_flagged_and_logged_not_raised(self):
        # Text says LMU but patient 10 sits in the TUM block: a miskeyed row.
        out, log = _run_cross_check("LMU", "definitive", 10)
        assert not bool(out.loc[0, "institution_matches_block"])
        assert len(log) == 1
        record = log.to_frame().iloc[0]
        assert record["column"] == "institution"

    def test_mojibaked_wuerzburg_still_canonicalises(self):
        # Wuerzburg patient 90 in the definitive block (88-92); the text arrives
        # with the classic cp1252/utf-8 double-decode residue.
        out, _ = _run_cross_check("WÃ¼rzburg", "definitive", 90)
        assert out.loc[0, "institution"] == "Wuerzburg"
        assert bool(out.loc[0, "institution_matches_block"])

    def test_unreadable_text_falls_back_to_the_block(self):
        out, _ = _run_cross_check("???", "neoadjuvant", 60)  # LMU block (58-67)
        assert out.loc[0, "institution"] == "LMU"


class TestCnInvolved:
    """`cn_involved` is the modelling covariate; `cn_stage_positive` is only the
    bare-'+' flag. Conflating them once put a near-constant column into the
    models, so the derivation is pinned down here."""

    def _derive(self, numbers, plus_flags):
        out = pd.DataFrame(
            {
                "cn_stage_num": pd.array(numbers, dtype="Int64"),
                "cn_stage_positive": plus_flags,
            }
        )
        raw = pd.DataFrame(
            {"cohort": ["definitive"] * len(out), "source_row": range(2, 2 + len(out))}
        )
        loading._add_derived_columns(out, raw, ParseLog())
        return out["cn_involved"]

    def test_counted_stages_map_to_involvement(self):
        involved = self._derive([0, 1, 2], [False, False, False])
        assert involved.tolist() == [False, True, True]

    def test_bare_plus_counts_as_involved(self):
        involved = self._derive([None], [True])
        assert involved.tolist() == [True]

    def test_missing_stage_without_plus_stays_missing(self):
        involved = self._derive([None], [False])
        assert involved.isna().tolist() == [True]
