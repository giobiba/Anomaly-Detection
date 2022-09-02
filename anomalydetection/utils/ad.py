import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def get_stats(sparse, y, CONTAMINATION):
    decision_scores = np.linalg.norm(sparse, ord=2, axis=1)

    threshold = pd.Series(decision_scores).quantile(1 - CONTAMINATION)

    y_pred = (decision_scores > threshold).astype(int)

    return get_stats_(y, y_pred)


def get_stats_(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prec, recall, fscore, support = precision_recall_fscore_support(y, y_pred)

    STATS = {
        'predicted labels': y_pred,
        'true negative': tn,
        'false positive': fp,
        'false negative': fn,
        'true positive': tp,
        'precision': prec[1],
        'recall': recall[1],
        'fscore': fscore[1]
    }
    return STATS