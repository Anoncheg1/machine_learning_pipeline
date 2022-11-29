import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels


def log_dtree(X: pd.DataFrame):
    preds = np.zeros(shape=(X.shape[0], 2), dtype=np.float)
    X.reset_index(inplace=True, drop=True)
    for i, r in X.iterrows():
        res = 0
        # m = DecisionTreeClassifier(class_weight={0: 1.7, 1: 1}, criterion='entropy',
        #                                max_depth=60, max_leaf_nodes=56, min_samples_leaf=7,
        #                                min_weight_fraction_leaf=0, random_state=7)
        # 1
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] > 745.5 and \
                r['CLIENT_WI_EXPERIENCE'] < 66.5 and \
                r['CLIENT_MARITAL_STATUS'] == 0 and \
                r['ANKETA_SCORING'] > 90.5 and \
                r['AUTO_DEAL_INITIAL_FEE'] <= 40500:
            res = 1
        # 2
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] > 808.5 and \
                r['CLIENT_WI_EXPERIENCE'] > 197.5:
            res = 1
        # 3
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] > 808.5:
            if r['CLIENT_WI_EXPERIENCE'] <= 197.5 and \
                    r['EQUIFAX_SCORING'] > 922:  # WHITE both
                res = 1
        # 4
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] > 808.5:
            if r['CLIENT_WI_EXPERIENCE'] > 197.5:  # WHITE
                if r['AUTO_DEAL_INITIAL_FEE'] > 760150:  # WHITE - more precisely
                    res = 1
        # ---------
        # 5
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] <= 745.5 and r['AUTO_DEAL_INITIAL_FEE'] <= 44500:
            if 500 < r['AUTO_DEAL_INITIAL_FEE'] <= 22500:  # white
                res = 1
        # 6
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                846.5 >= r['EQUIFAX_SCORING'] > 745.5 and r['AUTO_DEAL_INITIAL_FEE'] <= 48500:
            res = 1
        # 7
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                r['EQUIFAX_SCORING'] > 846.5 and \
                r['CLIENT_MARITAL_STATUS'] == 0:  # and r['OKB_SCORING'] <= 810:
            res = 1
        preds[i][0] = res <= 0
        preds[i][1] = res > 0
    return preds


def log_ranges(X: pd.DataFrame):
    preds = []
    for i, r in X.iterrows():
        res = 0
        # ---- 1 ------
        if 0.0 < r['CLIENT_WI_EXPERIENCE'] < 19.9 \
                and 79.4 < r['ANKETA_SCORING'] < 94.2 \
                and 937.0 < r['OKB_SCORING'] < 1068.0 \
                and 0 < r['CLIENT_MARITAL_STATUS'] < 1 \
                and r['OKB_RATING_SCORING_Хорошая КИ'] == 0 \
                and r['OKB_RATING_SCORING_КИ отсутствует'] == 1 \
                and r['NBKI_RATING_SCORING_Хорошая КИ'] == 0 \
                and r['NBKI_RATING_SCORING_КИ отсутствует'] == 1 \
                and r['МБКИ_тапоМБКИ'] == 0 \
                and 666.4 < r['EQUIFAX_SCORING'] < 761.6 \
                and 680.0 < r['NBKI_SCORING'] < 765.0 \
                and 190000.0 < r['AUTO_DEAL_INITIAL_FEE'] < 380000.0:
            res = 1
        preds.append(res > 0)
    return preds


def log_ranges2(X: pd.DataFrame):
    preds = []
    for i, r in X.iterrows():
        res = 0
        # ---- zz ------
        # CLIENT_MARITAL_STATUS {0: 'в браке', 1: 'не в браке'}
        # OKB_RATING_SCORING['Хорошая КИ' 'КИ отсутствует' 'Нейтральная КИ' 'Ошибка выполнения'  'Плохая КИ']
        # NBKI_RATING_SCORING ['Хорошая КИ' 'КИ отсутствует' 'Нейтральная КИ' 'Ошибка выполнения''Плохая КИ']
        # EQUIFAX_RATING_SCORING ['Хорошая КИ' 'КИ отсутствует' 'Ошибка выполнения' 'Плохая КИ' 'Нейтральная КИ']
        if (0.0 < r['CLIENT_WI_EXPERIENCE'] < 19.9 or 59.699999999999996 < r['CLIENT_WI_EXPERIENCE'] < 79.6) \
                and 79.4 < r['ANKETA_SCORING'] < 123.80000000000001 \
                and 675.0 < r['OKB_SCORING'] < 1068.0 \
                and r['CLIENT_MARITAL_STATUS'] == 0 \
                and r['OKB_RATING_SCORING_Хорошая КИ'] == 0 \
                and r['OKB_RATING_SCORING_КИ отсутствует'] == 1 \
                and 0 < r['NBKI_RATING_SCORING_Хорошая КИ'] < 1 \
                and r['NBKI_RATING_SCORING_КИ отсутствует'] == 1 \
                and 0 < r['МБКИ_тапоМБКИ'] < 1 \
                and 666.4 < r['EQUIFAX_SCORING'] < 761.6 \
                and 680.0 < r['NBKI_SCORING'] < 765.0 \
                and 190000.0 < r['AUTO_DEAL_INITIAL_FEE'] < 380000.0:
            res = 1
        preds.append(res > 0)
    return preds


# -- estimator
from sklearn.utils.multiclass import unique_labels


class MyEstim(BaseEstimator):
    def __init__(self, pred: callable):
        self.pred = pred
        self.classes_ = 2

    def fit(self, X=None, y=None):
        self.classes_ = unique_labels(y)
        pass

    def predict(self, X):
        return np.argmax(self.pred(X), axis=1)

    def score(self, X, y):
        y_pred = np.argmax(self.pred(X), axis=1)
        return metrics.accuracy_score(y, y_pred)

    def predict_proba(self, X):  # something wrong

        # z = np.array(list(zip(np.zeros(len(X))+0.1, self.pred(X))))
        # print(z[0], z[1])
        # z = np.argmax(, axis=1)
        return self.pred(X)
    # def set_params(self, **parameters):
    # for parameter, value in parameters.items():

    # def decision_function(self, X):
    #     z = np.array(list(zip(np.zeros(len(X)), self.pred(X))))
    #     print(z[0], z[1])
    #     return z

    # pred=log_ranges
    # pred = log_dtree
    # m = MyEstim(pred=pred)