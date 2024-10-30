from enum import Enum

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class BinaryMetricType(Enum):
    accuracy = BinaryAccuracy
    f1_score = BinaryF1Score
    auroc = BinaryAUROC
