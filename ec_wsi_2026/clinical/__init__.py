"""Clinical data curation for the esophageal cancer WSI cohort.

The two source workbooks (``definitive_clean.xlsx``, ``neoadjuvant_clean.xlsx``)
are hand-maintained across four institutions and are dirty in specific,
documented ways: the header sits on the second row under a colour legend, dates
are free text rather than Excel serials, stage columns carry substage letters,
and every duration column (PFS/OS/LFFS) is empty and must be derived from dates.

Pipeline::

    loading.load_cohorts()      raw workbooks -> one long all-string table
    coercion                    typed values + a logged trail of every failure
    endpoints.derive()          durations and event indicators for six endpoints
    audit.build_report()        population, consistency and exclusion tables

See ``references/endpoint_definitions.md`` for the time-origin and censoring
conventions, and ``references/code_legends.yaml`` for the coded-column meanings.
"""
