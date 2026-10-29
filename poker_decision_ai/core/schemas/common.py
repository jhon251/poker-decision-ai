"""Common shared schema fragments used across agents."""
from __future__ import annotations

#: Standard money/stack fields (always plain floats in bb or $; never formatted strings)
POSITIVE_NUMBER = {"type": "number", "minimum": 0.0}

#: A confidence value in [0, 1]
CONFIDENCE = {"type": "number", "minimum": 0.0, "maximum": 1.0}

#: A probability in [0, 1]
PROBABILITY = {"type": "number", "minimum": 0.0, "maximum": 1.0}

#: Standard evidence fragment attached to agent outputs
EVIDENCE_ARRAY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "text": {"type": "string"},
            "source": {"type": "string"},
            "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["kind", "text"],
    },
}

#: Standard warnings fragment
WARNINGS_ARRAY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "message": {"type": "string"}, "agent": {"type": "string"}},
        "required": ["code", "message"],
    },
}
