"""识别/失败状态（冻结 7 态）与下游阻断规则。

7 个状态：
  RECOGNITION_OK / RECOGNITION_UNCERTAIN / ROI_MISSING /
  LAYOUT_UNKNOWN / STATE_CONFLICT / FRAME_STALE / CAPTURE_FAILED

硬规则：STATE_CONFLICT（validator 判 invalid）必须硬阻断下游决策；
LAYOUT_UNKNOWN / FRAME_STALE / CAPTURE_FAILED 同样不可消费；
关键字段 uncertain 时 blocked=True（S10：关键 uncertainty 不进入下游）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecognitionStatus(str, Enum):
    RECOGNITION_OK = "RECOGNITION_OK"
    RECOGNITION_UNCERTAIN = "RECOGNITION_UNCERTAIN"
    ROI_MISSING = "ROI_MISSING"
    LAYOUT_UNKNOWN = "LAYOUT_UNKNOWN"
    STATE_CONFLICT = "STATE_CONFLICT"
    FRAME_STALE = "FRAME_STALE"
    CAPTURE_FAILED = "CAPTURE_FAILED"


#: 无条件硬阻断下游消费的状态
BLOCKING_STATUSES = frozenset({
    RecognitionStatus.STATE_CONFLICT,
    RecognitionStatus.LAYOUT_UNKNOWN,
    RecognitionStatus.FRAME_STALE,
    RecognitionStatus.CAPTURE_FAILED,
})


@dataclass(frozen=True)
class PipelineFailure:
    status: RecognitionStatus
    reason: str
    evidence: tuple = ()
    blocking: bool = True

    def to_dict(self) -> dict:
        return {"status": self.status.value, "reason": self.reason,
                "blocking": self.blocking, "evidence": list(self.evidence)}


def is_blocking_status(status: RecognitionStatus) -> bool:
    return status in BLOCKING_STATUSES
