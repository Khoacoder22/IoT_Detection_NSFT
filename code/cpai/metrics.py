"""Evaluation metrics matching the legacy `calc_index()` layout used in the Excel."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc as sk_auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)


def calc_index(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return (MCC, ACC, TPR_macro, FPR, PPV_macro, F1_macro, AUC, confusion_matrix).

    ACC and FPR are aggregated (micro) across classes: sum(TP+TN)/sum(all), sum(FP)/sum(FP+TN).
    TPR/PPV/F1 are macro-averaged. AUC is only computed for binary problems (else -1).
    Percentages are scaled x100 except MCC.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    acc = (tp + tn).sum() / (tp + fp + fn + tn).sum() * 100.0
    fpr = fp.sum() / (fp + tn).sum() * 100.0 if (fp + tn).sum() else 0.0

    tpr = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    ppv = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    mcc = matthews_corrcoef(y_true, y_pred)

    auc_val = -1.0
    if len(labels) == 2:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred)
        auc_val = sk_auc(fpr_curve, tpr_curve)

    return mcc, acc, tpr, fpr, ppv, f1, auc_val, cm


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same as calc_index but returns a dict keyed by the Excel column names."""
    mcc, acc, tpr, fpr, ppv, f1, auc_val, cm = calc_index(y_true, y_pred)
    return {
        "MCC": mcc,
        "ACC": acc,
        "TPR Macro": tpr,
        "FPR": fpr,
        "PPV Macro": ppv,
        "F1 Macro": f1,
        "AUC": auc_val,
        "CFS Matrix": cm,
    }


def evaluate_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Core metrics + F1 weighted + AUROC (OvR/OvO, macro+weighted) + per-class breakdown.

    `y_proba` should be shape (n_samples, n_classes) if given. If None, AUROC scores are skipped.
    """
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
    )

    mcc, acc, tpr, fpr, ppv, f1, auc_val, cm = calc_index(y_true, y_pred)

    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    ppv_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    tpr_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)

    result = {
        "MCC": mcc,
        "ACC": acc,
        "TPR Macro": tpr,
        "TPR Weighted": tpr_weighted,
        "FPR": fpr,
        "PPV Macro": ppv,
        "PPV Weighted": ppv_weighted,
        "F1 Macro": f1,
        "F1 Weighted": f1_weighted,
        "AUC (binary)": auc_val,
        "CFS Matrix": cm,
    }

    if y_proba is not None and n_classes >= 2:
        try:
            result["AUROC OvR Macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro") * 100.0
            result["AUROC OvR Weighted"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted") * 100.0
            result["AUROC OvO Macro"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro") * 100.0
        except Exception as e:
            result["AUROC Error"] = str(e)

        # Binary AUROC: collapse all non-benign classes into "attack"
        # Assumes benign class is index 0 ('0Normal' sorts first after LabelEncoder).
        if n_classes >= 2 and y_proba.shape[1] >= 2:
            try:
                y_true_bin = (y_true != 0).astype(int)
                attack_score = 1.0 - y_proba[:, 0]  # probability NOT benign
                result["AUROC Binary"] = roc_auc_score(y_true_bin, attack_score) * 100.0
                result["y_score_attack_binary"] = attack_score
            except Exception as e:
                result["AUROC Binary Error"] = str(e)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    result["Per-class report"] = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=class_names[:n_classes] if len(class_names) >= n_classes else None,
        output_dict=True, zero_division=0,
    )
    return result
