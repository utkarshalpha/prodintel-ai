"""Live Analysis mode -- run the real pipeline on user input with the local engine.

``runtime`` is the Streamlit-free execution layer (unit-testable anywhere); ``view`` is the
Streamlit UI. Both reuse L4 ingestion, PipelineService, LocalHeuristicClient,
assemble_snapshot, and the existing renderers -- no logic is duplicated.
"""
