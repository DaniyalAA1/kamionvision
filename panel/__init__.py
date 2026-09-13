"""The condition-reference panel.

`app/` appraises. `panel/` builds the thing an appraisal can be measured
against: a condition reference for corpus vehicles, graded by a panel of
independent Claude Opus 5 agents under the rubric that lives in
`app/evidence/prompts.py`.

It is a silver standard, not ground truth, and it is deliberately not under
`app/` - nothing in the appraisal path imports it. It is a data-collection
instrument that writes an artifact (`data/reference/condition_panel_v1.jsonl`)
and a card that says how the artifact was made.
"""
