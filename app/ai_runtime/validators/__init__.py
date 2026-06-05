"""Stage-specific semantic validators (e.g. grounding, evidence-completeness).

Each validator implements :class:`app.ai_runtime.interfaces.StageValidator` and is
injected into the relevant stage runner. Kept in their own package so the harness
core stays free of stage-specific logic.
"""
