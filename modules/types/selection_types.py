from enum import Enum


class SelectRejectMode(Enum):
    CONFIDENCE = 1
    ENTROPY = 2
    MARGIN = 3
    WEIGHTED = 4
    TOPS = 5
