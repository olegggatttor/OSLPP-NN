from enum import Enum


class SelectRejectMode(Enum):
    CONFIDENCE = 1
    ENTROPY = 2
    MARGIN = 3
    WEIGHTED = 4
    TOPS = 5

    CONFIDENCE_MULTI = 6
    MARGIN_MULTI = 7
    TOTAL_U = 8
    WEIGHTED_MULTI = 9
    TOPS_MULTI = 10

    AUGMENTATION_SIMILARITY = 11
