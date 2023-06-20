from enum import Enum


class Metrics(Enum):
    """
    Enum class for metrics
    """
    bal_acc = 'Balanced Accuracy'
    f1_score = 'F1-Score'
    precision = 'Precision'
    recall = 'Recall'
