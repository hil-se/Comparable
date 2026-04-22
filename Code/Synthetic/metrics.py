import math

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import t, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler


class Metrics:
    def __init__(self, y, y_pred):
        self.y = np.asarray(y)
        self.y_pred = np.asarray(y_pred)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.y, self.y_pred)

    def f1(self):
        return sklearn.metrics.f1_score(self.y, self.y_pred)

    def precision(self):
        return sklearn.metrics.precision_score(self.y, self.y_pred)

    def recall(self):
        return sklearn.metrics.recall_score(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def pearsonr_coefficient(self):
        return pearsonr(self.y, self.y_pred)[0]

    def pearsonr_value(self):
        return pearsonr(self.y, self.y_pred)[1]

    def spearmanr_coefficient(self):
        return spearmanr(self.y, self.y_pred)[0]

    def spearmanr_value(self):
        return spearmanr(self.y, self.y_pred)[1]

    def confusion(self, y, y_pred):
        y = np.asarray(y) > 0
        y_pred = np.asarray(y_pred) > 0
        tp = np.sum(y & y_pred)
        fp = np.sum(~y & y_pred)
        tn = np.sum(~y & ~y_pred)
        fn = np.sum(y & ~y_pred)
        return tp, fp, tn, fn

    @staticmethod
    def _safe_div(num, den):
        return float(num) / den if den != 0 else 0.0

    def _group_confusion(self, s, value):
        mask = np.asarray(s) == value
        return self.confusion(self.y[mask], self.y_pred[mask])

    def _ordered_pair_error_diff(self, s):
        s = np.asarray(s)
        total = 0.0
        count = 0
        for i in range(len(s)):
            for j in range(len(s)):
                if s[i] > s[j]:
                    total += (self.y_pred[i] - self.y_pred[j]) - (self.y[i] - self.y[j])
                    count += 1
        return total, count

    def _linear_predictions(self, s):
        joint = pd.DataFrame({"y": self.y, "y_pred": self.y_pred})
        margin = self.y.reshape(-1, 1)
        pred_joint = LinearRegression().fit(joint, s).predict(joint)
        pred_margin = LinearRegression().fit(margin, s).predict(margin)
        return pred_joint, pred_margin

    @staticmethod
    def _comparative_counts(y, y_pred, groups):
        groups = np.asarray(groups, dtype=int)
        y_pos = np.asarray(y) == 1
        pred_pos = np.asarray(y_pred) == 1
        return {
            "tp": np.bincount(groups, weights=(y_pos & pred_pos), minlength=4),
            "fn": np.bincount(groups, weights=(y_pos & ~pred_pos), minlength=4),
            "fp": np.bincount(groups, weights=(~y_pos & pred_pos), minlength=4),
            "tn": np.bincount(groups, weights=(~y_pos & ~pred_pos), minlength=4),
        }

    @staticmethod
    def _mi_term(count, left_total, right_total, global_total, normalizer):
        if not count or not left_total or not right_total or not global_total:
            return 0.0
        return (
            np.log((count / left_total) / (right_total / global_total))
            * count
            / normalizer
        )

    def EOD(self, s):
        tp, fp, tn, fn = self._group_confusion(s, 0)
        op0 = self._safe_div(tp, tp + fn)
        tp, fp, tn, fn = self._group_confusion(s, 1)
        op1 = self._safe_div(tp, tp + fn)
        return op1 - op0

    def AOD(self, s):
        tp, fp, tn, fn = self._group_confusion(s, 0)
        od0 = self._safe_div(tp, tp + fn) + self._safe_div(fp, fp + tn)
        tp, fp, tn, fn = self._group_confusion(s, 1)
        od1 = self._safe_div(tp, tp + fn) + self._safe_div(fp, fp + tn)
        return (od1 - od0) / 2

    def RBD(self, s):
        if len(np.unique(s)) == 2:
            errors = self.y_pred - self.y
            s = np.asarray(s)
            bias_diff = np.mean(errors[s == 1]) - np.mean(errors[s == 0])
        else:
            total, count = self._ordered_pair_error_diff(s)
            bias_diff = self._safe_div(total, count)
        sigma = np.std(self.y_pred - self.y, ddof=1)
        return bias_diff / sigma if sigma else 0.0

    def RBT(self, s):
        if len(np.unique(s)) == 2:
            errors = self.y_pred - self.y
            s = np.asarray(s)
            group1 = errors[s == 1]
            group0 = errors[s == 0]
            bias_diff = np.mean(group1) - np.mean(group0)
            var1 = np.var(group1, ddof=1)
            var0 = np.var(group0, ddof=1)
            var = var1 / len(group1) + var0 / len(group0)
            if var > 0:
                bias_diff = bias_diff / np.sqrt(var)
                dof = var**2 / (
                    (var1 / len(group1)) ** 2 / (len(group1) - 1)
                    + (var0 / len(group0)) ** 2 / (len(group0) - 1)
                )
            else:
                bias_diff = 0.0
                dof = 1
        else:
            total, count = self._ordered_pair_error_diff(s)
            bias_diff = self._safe_div(total, count)
            sigma = np.std(self.y_pred - self.y, ddof=1)
            if sigma:
                bias_diff = bias_diff * np.sqrt(len(s)) / sigma
            else:
                bias_diff = 0.0
            dof = len(s) - 1
        p = t.sf(np.abs(bias_diff), dof)
        return p

    def r_sep(self, s):
        joint = pd.DataFrame(
            {"y": self.y, "y_pred": self.y_pred}, columns=["y", "y_pred"]
        )
        margin = self.y.reshape(-1, 1)
        model_joint = LogisticRegression().fit(joint, s)
        model_margin = LogisticRegression().fit(margin, s)

        prob_joint = model_joint.predict_proba(joint)[:, 1]
        prob_margin = model_margin.predict_proba(margin)[:, 1]
        ratio = 0

        for i in range(len(s)):
            t = (prob_joint[i] / (1 - prob_joint[i])) * (
                (1 - prob_margin[i]) / prob_margin[i]
            )
            ratio = ratio + t
        ratio = ratio / len(s)
        return ratio

    def MI(self, s):
        joint = pd.DataFrame(
            {"y": self.y, "y_pred": self.y_pred}, columns=["y", "y_pred"]
        )
        margin = self.y.reshape(-1, 1)
        model_joint = LogisticRegression().fit(joint, s)
        model_margin = LogisticRegression().fit(margin, s)

        prob_joint = model_joint.predict_proba(joint)
        prob_margin = model_margin.predict_proba(margin)
        Info = 0
        Entropy = 0

        for i in range(len(s)):
            Info = Info + math.log(prob_joint[i][s[i]] / prob_margin[i][s[i]])
            Entropy = Entropy + math.log(prob_margin[i][s[i]])

        MI = Info / (-Entropy)
        return MI

    def MI_b(self, s):
        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp0, fp0, tn0, fn0 = self.confusion(y0, y0_pred)
        tp1, fp1, tn1, fn1 = self.confusion(y1, y1_pred)

        def ediff(n1, d1, n0, d0):
            return (
                np.log(n0 / (n0 + d0)) * n0
                + np.log(n1 / (n1 + d1)) * n1
                - (n0 + n1) * np.log((n0 + n1) / (n0 + n1 + d0 + d1))
            )

        MI = (
            ediff(tp1, fn1, tp0, fn0)
            + ediff(fp1, tn1, fp0, tn0)
            + ediff(tn1, fp1, tn0, fp0)
            + ediff(fn1, tp1, fn0, tp0)
        )
        return MI / len(s)

    def MI_con(self, s):
        pred_joint, pred_margin = self._linear_predictions(s)
        eps = np.finfo(float).eps
        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)
        rse_joint = max(rse_joint, eps)
        rse_margin = max(rse_margin, eps)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)
        info = np.log(pdf_joint / pdf_margin).sum()
        entropy = np.log(pdf_margin).sum()
        return info / (-entropy)

    def MI_con_scaled(self, s):
        pred_joint, pred_margin = self._linear_predictions(s)
        eps = np.finfo(float).eps
        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)
        rse_joint = max(rse_joint, eps)
        rse_margin = max(rse_margin, eps)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)

        concat_pdf = np.concatenate((pdf_joint, pdf_margin))

        scaler = MinMaxScaler(feature_range=(0.01, 0.99))

        scaled_concat_pdf = scaler.fit_transform(concat_pdf.reshape(-1, 1))

        scaled_joint_pdf, scaled_pdf_margin = np.array_split(scaled_concat_pdf, 2)

        info = np.log(scaled_joint_pdf / scaled_pdf_margin).sum()
        entropy = np.log(scaled_pdf_margin).sum()
        return info / (-entropy)

    def MI_con_info(self, s):
        pred_joint, pred_margin = self._linear_predictions(s)
        eps = np.finfo(float).eps
        rse_joint = max(np.std(pred_joint - s), eps)
        rse_margin = max(np.std(pred_margin - s), eps)

        pdf_joint = np.clip(norm.pdf(s, pred_joint, rse_joint), eps, None)
        pdf_margin = np.clip(norm.pdf(s, pred_margin, rse_margin), eps, None)
        return np.log(pdf_joint / pdf_margin).sum() / len(s)

    def AOD_comp(self, s):
        direction = np.sign(s["A"].to_numpy() - s["B"].to_numpy())
        mask = direction != 0
        y = self.y[mask] * direction[mask]
        y_pred = self.y_pred[mask] * direction[mask]
        tp, fp, tn, fn = self.confusion(y, y_pred)
        t = tp + fn
        n = fp + tn
        tpr = self._safe_div(tp, t)
        tnr = self._safe_div(tn, n)
        fpr = self._safe_div(fp, n)
        fnr = self._safe_div(fn, t)
        return (tpr + fpr - tnr - fnr) / 2

    def Within_comp(self, s):
        same_one = (s["A"].to_numpy() == 1) & (s["B"].to_numpy() == 1)
        same_zero = (s["A"].to_numpy() == 0) & (s["B"].to_numpy() == 0)
        acc_one = (
            np.mean(self.y[same_one] == self.y_pred[same_one])
            if np.any(same_one)
            else 0.0
        )
        acc_zero = (
            np.mean(self.y[same_zero] == self.y_pred[same_zero])
            if np.any(same_zero)
            else 0.0
        )
        return acc_one - acc_zero

    def Sep_comp(self, s):
        return np.sqrt(self.Within_comp(s) ** 2 + self.AOD_comp(s) ** 2)

    def gAOD(self, s):
        t = n = tp = fp = tn = fn = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] - s[j] > 0:
                    if self.y[i] - self.y[j] > 0:
                        t += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            tp += 1
                        if self.y_pred[i] < self.y_pred[j]:
                            fn += 1
                    elif self.y[j] - self.y[i] > 0:
                        n += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            fp += 1
                        elif self.y_pred[i] < self.y_pred[j]:
                            tn += 1

        tpr = self._safe_div(tp, t)
        tnr = self._safe_div(tn, n)
        fpr = self._safe_div(fp, n)
        fnr = self._safe_div(fn, t)
        return (tpr + fpr - tnr - fnr) / 2

    def gWithin(self, s):
        correct1 = total1 = correct0 = total0 = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] == s[j] == 1:
                    true_sign = np.sign(self.y[i] - self.y[j])
                    pred_sign = np.sign(self.y_pred[i] - self.y_pred[j])
                    if true_sign != 0:
                        total1 += 1
                        correct1 += pred_sign == true_sign
                elif s[i] == s[j] == 0:
                    true_sign = np.sign(self.y[i] - self.y[j])
                    pred_sign = np.sign(self.y_pred[i] - self.y_pred[j])
                    if true_sign != 0:
                        total0 += 1
                        correct0 += pred_sign == true_sign

        return self._safe_div(correct1, total1) - self._safe_div(correct0, total0)

    def gSep(self, s):
        return np.sqrt(self.gWithin(s) ** 2 + self.gAOD(s) ** 2)

    def MI_comp(self, s):
        groups = (s["A"].to_numpy(dtype=int) << 1) + s["B"].to_numpy(dtype=int)
        counts = self._comparative_counts(self.y, self.y_pred, groups)
        t = counts["tp"] + counts["fn"]
        n = counts["fp"] + counts["tn"]
        tp = counts["tp"].sum()
        fn = counts["fn"].sum()
        fp = counts["fp"].sum()
        tn = counts["tn"].sum()
        normalizer = len(s)

        return sum(
            self._mi_term(counts["tp"][group], tp, t[group], t.sum(), normalizer)
            + self._mi_term(counts["fn"][group], fn, t[group], t.sum(), normalizer)
            + self._mi_term(counts["fp"][group], fp, n[group], n.sum(), normalizer)
            + self._mi_term(counts["tn"][group], tn, n[group], n.sum(), normalizer)
            for group in range(4)
        )

    def MI_comp2(self, s):
        groups = (s["A"].to_numpy(dtype=int) << 1) + s["B"].to_numpy(dtype=int)
        counts = self._comparative_counts(self.y, self.y_pred, groups)
        t = counts["tp"] + counts["fn"]
        n = counts["fp"] + counts["tn"]
        tp = counts["tp"].sum()
        fn = counts["fn"].sum()
        fp = counts["fp"].sum()
        tn = counts["tn"].sum()
        normalizer = len(s)

        return sum(
            self._mi_term(counts["tp"][group], t[group], tp, t.sum(), normalizer)
            + self._mi_term(counts["fn"][group], t[group], fn, t.sum(), normalizer)
            + self._mi_term(counts["fp"][group], n[group], fp, n.sum(), normalizer)
            + self._mi_term(counts["tn"][group], n[group], tn, n.sum(), normalizer)
            for group in range(4)
        )
