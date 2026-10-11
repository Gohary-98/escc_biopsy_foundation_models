"""Header-resolution tests: the one place a renamed workbook column can vanish.

`resolve_columns` matches workbook headers against regex patterns. If a future
export renames a column, the canonical name must drop OUT of the mapping and the
new header must show UP in `unmatched` — silence in either direction is how a
covariate disappears without anyone noticing.
"""

from __future__ import annotations

from ec_wsi_2026.clinical import schema as sch


class TestResolveColumns:
    def test_known_headers_resolve(self):
        mapping, unmatched = sch.resolve_columns(
            ["Patient No", "Institution", "Date Start CRT", "Progress", "Death"]
        )
        assert mapping["patient_no"] == "Patient No"
        assert mapping["date_start_crt"] == "Date Start CRT"
        assert unmatched == []

    def test_renamed_header_is_dropped_and_reported(self):
        # The drift scenario: someone renames "Date Start CRT" upstream. The
        # canonical must NOT silently bind to the stranger, and the stranger
        # must be reported so the audit surfaces it.
        mapping, unmatched = sch.resolve_columns(["Therapy Commencement Day"])
        assert "date_start_crt" not in mapping
        assert "Therapy Commencement Day" in unmatched

    def test_a_header_is_claimed_at_most_once(self):
        # Two canonicals must never bind the same source header: first claim
        # wins and the pattern search skips claimed headers.
        headers = ["Date Start CRT", "Date End CRT"]
        mapping, _ = sch.resolve_columns(headers)
        claimed = list(mapping.values())
        assert len(claimed) == len(set(claimed))

    def test_deliberately_ignored_columns_are_not_reported(self):
        # The empty duration columns are known junk; they must not pollute the
        # unmatched report that a human is expected to read.
        _, unmatched = sch.resolve_columns(["PFS (days)", "OS (days)"])
        assert unmatched == []
