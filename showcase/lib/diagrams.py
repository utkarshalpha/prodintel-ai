"""Architecture-mode diagrams -- the five approved Mermaid sources (no new diagrams).

These are the diagrams authored in the architecture-asset step, kept verbatim as the
source of truth for Architecture mode. Each entry renders as a card (rendered diagram +
a "view source" expander).
"""

from __future__ import annotations

__all__ = ["DIAGRAMS"]

_SYSTEM = """flowchart TD
    Client(["Client / Recruiter"])
    subgraph API["API -- FastAPI"]
        Routes["Routes + Pydantic schemas<br/>correlation-id middleware"]
    end
    subgraph SVC["Service Layer -- 3-phase txn (Read - AI - Write)"]
        OTHERS["SignalService / FeatureService / ConflictService"]
        S4S["DecisionService"]
        WHY["DecisionExplanationService"]
    end
    subgraph STAGES["AI Stages"]
        ST1["Stage 1 - Signal Analysis"]
        ST2["Stage 2 - Feature Extraction"]
        ST3["Stage 3 - Conflict Detection"]
        ST4["Stage 4 - Decision Synthesis"]
    end
    subgraph HARNESS["Runtime Harness"]
        RUNNER["BaseStageRunner / RetryPolicy / StageResult"]
        GATES{{"Validation Gates:<br/>grounding / provenance /<br/>conflict-integrity / decision-integrity"}}
    end
    subgraph RAG["Retrieval Layer (opt-in)"]
        RS["RetrievalService"]
        VI["Embeddings + Vector index"]
    end
    GROUND["Framework Grounding<br/>frozen citation pool to Stage 4"]
    LLM[["Claude -- ToolCallClient adapter<br/>(single, swappable LLM boundary)"]]
    subgraph STORE["Persistence"]
        PG[("PostgreSQL<br/>12 tables + provenance edges")]
        CH[("ChromaDB<br/>optional - fake by default")]
    end
    Client --> Routes
    Routes --> OTHERS
    Routes --> S4S
    Routes --> WHY
    OTHERS --> ST1
    OTHERS --> ST2
    OTHERS --> ST3
    S4S --> ST4
    ST1 --> RUNNER
    ST2 --> RUNNER
    ST3 --> RUNNER
    ST4 --> RUNNER
    RUNNER --> GATES
    RUNNER --> LLM
    S4S -. opt-in .-> RS
    RS --> VI
    RS --> GROUND
    GROUND --> ST4
    VI -.-> CH
    OTHERS --> PG
    S4S --> PG
    WHY --> PG
"""

_PIPELINE = """flowchart TD
    SIG["Stakeholder Signals<br/>immutable - deduped by content hash"]
    A["Stage 1 - Signal Analysis<br/>gate: grounding"]
    F["Stage 2 - Feature Extraction<br/>gate: provenance"]
    C["Stage 3 - Conflict Detection<br/>gate: conflict integrity"]
    subgraph S4["Stage 4 - Decision Synthesis"]
        R["Framework Retrieval<br/>(Phase 1.5 - opt-in)"]
        POOL["Frozen framework pool"]
        SYN["Synthesize<br/>gates: decision-integrity + citation grounding"]
    end
    GD["Grounded, evidence-backed Decisions"]
    P[("Persistence -- FK-protected edges")]
    WHY[["GET /decisions/:id/why"]]
    SIG --> A --> F --> C --> R
    R --> POOL --> SYN --> GD --> P --> WHY
"""

_PROVENANCE = """flowchart TD
    D["Decision<br/>recommendation - rank - rationale"]
    EV["Evidence Signals<br/>(decision_evidence - FK RESTRICT)"]
    CF["Acknowledged Conflicts<br/>(decision_conflict - FK RESTRICT)"]
    FC["Framework Citations<br/>(decision_framework_citation - GROUNDS)"]
    FT["Subject Feature"]
    SIG["Stakeholder Signals"]
    KC["Knowledge Chunks<br/>(framework corpus)"]
    TXT["Original stakeholder text<br/>claim + character span (quoted_text)"]
    D --> EV
    D --> CF
    D --> FC
    D --> FT
    EV --> SIG
    CF --> SIG
    FT --> SIG
    FC --> KC
    SIG --> TXT
"""

_TRUST = """flowchart TD
    OUT["LLM tool output<br/>(untrusted)"]
    SCHEMA{"Schema validation<br/>Pydantic strict - extra=forbid"}
    SEM{"Decision-integrity gate<br/>real subjects / evidence /<br/>every conflict acknowledged"}
    CITE{"Framework-citation validation<br/>every cited chunk in the retrieved pool"}
    PERSIST[("Persist -- Postgres")]
    RETRY["Deterministic retry feedback<br/>(exact violated constraint)"]
    OUT --> SCHEMA
    SCHEMA -- pass --> SEM
    SEM -- pass --> CITE
    CITE -- pass --> PERSIST
    SCHEMA -- fail --> RETRY
    SEM -- fail --> RETRY
    CITE -- fail --> RETRY
    RETRY --> OUT
"""

_RETRIEVAL = """flowchart TD
    FEAT["Feature<br/>(title + jtbd)"]
    Q["Retrieval Query<br/>(pinned to corpus_version)"]
    E["Embedding<br/>(fake by default - sentence-transformers optional)"]
    VS["Vector Search<br/>(top-k - cosine)"]
    KC["Knowledge Chunks<br/>(text resolved from Postgres)"]
    POOL["Framework Pool<br/>(dedupe by chunk_id - keep max score -<br/>frozen, deterministic order)"]
    PROMPT["Stage 4 Prompt<br/>(+ citation-grounding gate)"]
    VI[("Vector index<br/>ChromaDB / in-memory fake")]
    PG[("PostgreSQL<br/>chunk text")]
    FEAT --> Q --> E --> VS --> KC --> POOL --> PROMPT
    VS -.-> VI
    KC -.-> PG
"""

DIAGRAMS = [
    {"key": "system", "title": "System Architecture",
     "description": "Layered & dependency-injected: HTTP - Services - Stages/Harness - Validators/RAG - DB.",
     "mermaid": _SYSTEM, "height": 640},
    {"key": "pipeline", "title": "Decision Pipeline",
     "description": "Signals - analysis - features - conflicts - grounded decision - persistence - /why.",
     "mermaid": _PIPELINE, "height": 520},
    {"key": "provenance", "title": "Provenance Graph",
     "description": "Every decision traces back to evidence, conflicts, framework citations, and original text.",
     "mermaid": _PROVENANCE, "height": 480},
    {"key": "trust", "title": "Trust Boundary",
     "description": "The model is untrusted: output passes schema + integrity + citation gates or is rejected.",
     "mermaid": _TRUST, "height": 460},
    {"key": "retrieval", "title": "Retrieval Architecture",
     "description": "Per-feature retrieval builds one frozen, deterministically-ordered framework pool.",
     "mermaid": _RETRIEVAL, "height": 460},
]
