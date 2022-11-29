import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels


def log_xgtree_orig(X: pd.DataFrame):
    preds = np.zeros(shape=(X.shape[0], 2), dtype=np.float)
    X.reset_index(inplace=True, drop=True)
    for i, r in X.iterrows():
        res = []
        # -- 1
        if r['OKB_RATING_SCORING_КИ отсутствует'] == 1:
            res.append(-0.45)  # higher - higher precision
        elif r['OKB_RATING_SCORING_Хорошая КИ'] == 1:
            res.append(0.1)  # lower higher precision
        else:
            res.append(0.05)

        # -- 2
        res.append((r['EQUIFAX_SCORING'] * 0.6) / 255 - 2.2)

        # -- 3
        # low precision
        # x = r['CLIENT_WI_EXPERIENCE']
        # a1, b1 = 0.11162212531024945, 0.02
        # a2, b2 = 0.2232442506204989, 0.02
        # if r['OKB_RATING_SCORING_КИ отсутствует'] == 0:
        #     y = a1 * np.log(b1 * x)
        # else:
        #     y = a2 * np.log(b2 * x + 0.2)
        # res +=y
        # low recall
        if r['CLIENT_WI_EXPERIENCE'] < 80:
            res.append(-0.3)  # higher - higher precision
        else:
            res.append(0.1)  # lower - higher precision
        # -- 4
        # {0: 'в браке', 1: 'не в браке'}
        if r['CLIENT_MARITAL_STATUS'] == 1:
            res.append(-0.15)
        else:
            res.append(0.02)
        # -- 5
        if 0 <= r['AUTO_DEAL_INITIAL_FEE'] < 50000:
            res.append(0.2)
        elif 50000 <= r['AUTO_DEAL_INITIAL_FEE'] < 300000:
            res.append(- 0.02)
        elif 300000 > r['AUTO_DEAL_INITIAL_FEE']:
            res.append(0.1)
        preds[i][0] = - sum([x for x in res if x <= 0])
        preds[i][1] = sum([x for x in res if x > 0])
    return preds


class MyEstimXGB_orig(BaseEstimator):
    def __init__(self, v1a=0, v1b=0, v1c=0, v2a=0, v2b=0,
                 v3a=0, v3b=0, v3c=0, v4a=0, v4b=0, v5a=0, v5b=0, v5c=0, v6a=0, v6b=0, v6c=0,
                 v7a=0, v7b=0, v71a=0, v71b=0, v72a=0, v72b=0, v73a=0, v73b=0):
        self.v1a = v1a
        self.v1b = v1b
        self.v1c = v1c
        self.v2a = v2a
        self.v2b = v2b
        self.v3a = v3a
        self.v3b = v3b
        self.v3c = v3c
        self.v4a = v4a
        self.v4b = v4b
        self.v5a = v5a
        self.v5b = v5b
        self.v5c = v5c
        self.v6a = v6a  # NBKI scoring
        self.v6b = v6b
        self.v6c = v6c
        self.v7a = v7a  # MBKI scoring factors # multiplier
        self.v7b = v7b  # MBKI scoring factors # multiplier
        # self.v71a = v71a
        self.v71b = v71b
        # self.v72a = v72a
        self.v72b = v72b
        # self.v73a = v73a
        self.v73b = v73b
        # for sklearn
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

    def get_params(self, deep=True):
        return {
            "v1a": self.v1a,
            "v1b": self.v1b,
            "v1c": self.v1c,
            "v2a": self.v2a,
            "v2b": self.v2b,
            "v3a": self.v3a,
            "v3b": self.v3b,
            "v3c": self.v3c,
            "v4a": self.v4a,
            "v4b": self.v4b,
            "v5a": self.v5a,
            "v5b": self.v5b,
            "v5c": self.v5c,
            "v6a": self.v6a,
            "v6b": self.v6b,
            "v6c": self.v6c,
            "v7a": self.v7a,
            "v7b": self.v7b,
            # "v71a": self.v71a,
            "v71b": self.v71b,
            # "v72a": self.v72a,
            "v72b": self.v72b,
            # "v73a": self.v73a,
            "v73b": self.v73b
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # self.params_est = parameters
        return self

    # for parameter, value in parameters.items():

    # def decision_function(self, X):
    #     z = np.array(list(zip(np.zeros(len(X)), self.pred(X))))
    #     print(z[0], z[1])
    #     return z

    def pred(self, X: pd.DataFrame):
        preds = np.zeros(shape=(X.shape[0], 2), dtype=np.float)
        X.reset_index(inplace=True, drop=True)
        for i, r in X.iterrows():
            # print(r.tolist())
            res = []
            # -- 1
            if r['OKB_RATING_SCORING_КИ отсутствует'] == 1:
                res.append(self.v1a)  # higher - higher precision
            elif r['OKB_RATING_SCORING_Хорошая КИ'] == 1:
                res.append(self.v1b)  # lower higher precision
            else:
                res.append(self.v1c)

            # -- 2 a b c
            res.append(r['EQUIFAX_SCORING'] * self.v2a - self.v2b)

            # -- 3 a b c
            # low precision
            # x = r['CLIENT_WI_EXPERIENCE']
            # a1, b1 = 0.11162212531024945, 0.02
            # a2, b2 = 0.2232442506204989, 0.02
            # if r['OKB_RATING_SCORING_КИ отсутствует'] == 0:
            #     y = a1 * n~p.log(b1 * x)
            # else:
            #     y = a2 * np.log(b2 * x + 0.2)
            # res +=y
            # low recall
            if r['CLIENT_WI_EXPERIENCE'] < self.v3a:
                res.append(self.v3b)  # higher - higher precision
            else:
                res.append(self.v3c)  # lower - higher precision
            # -- 4 a b
            # {0: 'в браке', 1: 'не в браке'}
            if r['CLIENT_MARITAL_STATUS'] == 1:
                res.append(self.v4a)
            else:
                res.append(self.v4b)
            # -- 5 a b c
            if 0 <= r['AUTO_DEAL_INITIAL_FEE'] < 50000:
                res.append(self.v5a)
            elif 50000 <= r['AUTO_DEAL_INITIAL_FEE'] < 300000:
                res.append(self.v5b)
            elif 300000 > r['AUTO_DEAL_INITIAL_FEE']:
                res.append(self.v5c)
            # -- 6
            if r['NBKI_SCORING'] <= 700:
                res.append(self.v6c)
            else:
                res.append(r['NBKI_SCORING'] * self.v6a - self.v6b)
            # -- 7
            if r['OKB_RATING_SCORING_КИ отсутствует']:
                multiplier = self.v7a * r['OKB_RATING_SCORING_КИ отсутствует']
            else:
                multiplier = self.v7b * r['OKB_RATING_SCORING_КИ отсутствует']
            # if r['МБКИ_требаналотч'] == 0:
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
            #     res.append(self.v71b + self.v71b * multiplier)
            #
            # if r['МБКИ_треб_исп_пр'] == 0:
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
            #     res.append(self.v72b + self.v72b * multiplier)
            #
            # if r['МБКИ_нет_огр'] == 0:
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
            #     res.append(self.v73b + self.v73b * multiplier)
            if r['МБКИ_требаналотч'] == 1:
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
                res.append(self.v71b * multiplier)

            if r['МБКИ_треб_исп_пр'] == 1:
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
                res.append(self.v72b * multiplier)

            if r['МБКИ_нет_огр'] == 1:
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
                res.append(self.v73b * multiplier)

            preds[i][0] = - sum([x for x in res if x <= 0])
            preds[i][1] = sum([x for x in res if x > 0])
        return preds

# -------------------------------------------------------------------------------------------


class MyEstimXGB(BaseEstimator):
    def __init__(self, v1a=0, v1b=0, v1c=0, v2a=0, v2b=0,
                 v3a=0, v3b=0, v3c=0, v4a=0, v4b=0, v5a=0, v5b=0, v5c=0, v6a=0, v6b=0, v6c=0,
                 v7a=0, v7b=0, v71a=0, v71b=0, v72a=0, v72b=0, v73a=0, v73b=0):
        self.v1a = v1a
        self.v1b = v1b
        self.v1c = v1c
        self.v2a = v2a
        self.v2b = v2b
        self.v3a = v3a
        self.v3b = v3b
        self.v3c = v3c
        self.v4a = v4a
        self.v4b = v4b
        self.v5a = v5a
        self.v5b = v5b
        self.v5c = v5c
        self.v6a = v6a  # NBKI scoring
        self.v6b = v6b
        self.v6c = v6c
        self.v7a = v7a  # MBKI scoring factors # multiplier
        self.v7b = v7b  # MBKI scoring factors # multiplier
        # self.v71a = v71a
        self.v71b = v71b
        # self.v72a = v72a
        self.v72b = v72b
        # self.v73a = v73a
        self.v73b = v73b
        # for sklearn
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

    def get_params(self, deep=True):
        return {
            "v1a": self.v1a,
            "v1b": self.v1b,
            "v1c": self.v1c,
            "v2a": self.v2a,
            "v2b": self.v2b,
            "v3a": self.v3a,
            "v3b": self.v3b,
            "v3c": self.v3c,
            "v4a": self.v4a,
            "v4b": self.v4b,
            "v5a": self.v5a,
            "v5b": self.v5b,
            "v5c": self.v5c,
            "v6a": self.v6a,
            "v6b": self.v6b,
            "v6c": self.v6c,
            "v7a": self.v7a,
            "v7b": self.v7b,
            # "v71a": self.v71a,
            "v71b": self.v71b,
            # "v72a": self.v72a,
            "v72b": self.v72b,
            # "v73a": self.v73a,
            "v73b": self.v73b
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # self.params_est = parameters
        return self

    # for parameter, value in parameters.items():

    # def decision_function(self, X):
    #     z = np.array(list(zip(np.zeros(len(X)), self.pred(X))))
    #     print(z[0], z[1])
    #     return z

    def pred(self, X: pd.DataFrame):
        preds = np.zeros(shape=(X.shape[0], 2), dtype=np.float)
        X = X.reset_index(drop=True).copy()
        # for i, r in X.iterrows():
        from transliterate import translit
        # X.columns = [translit(v, 'ru', reversed=True) for v in X.columns]
        # -- UNCOMMENT TO ENSURE RIGHT ORDER
        # print([(i, v) for i, v in enumerate(X.columns)])
        orig_cols = [(0, 'CLIENT_AGE'), (1, 'CLIENT_MARITAL_STATUS'), (2, 'CLIENT_DEPENDENTS_COUNT'), (3, 'CLIENT_WI_EXPERIENCE'), (4, 'ANKETA_SCORING'), (5, 'OKB_SCORING'), (6, 'EQUIFAX_SCORING'), (7, 'NBKI_SCORING'), (8, 'AUTO_DEAL_INITIAL_FEE'), (9, 'МБКИ_треб_адрес'), (10, 'МБКИ_треб_исп_пр'), (11, 'МБКИ_треб_пассп'), (12, 'МБКИ_требаналотч'), (13, 'МБКИ_нет_огр'), (14, 'МБКИ_недост'), (15, 'МБКИ_розыск'), (16, 'МБКИ_невыполнена'), (17, 'МБКИ_налспецуч'), (18, 'МБКИ_данные_не'), (19, 'День недели'), (20, 'OKB_RATING_SCORING_КИ отсутствует'), (21, 'OKB_RATING_SCORING_Хорошая КИ'), (22, 'OKB_RATING_SCORING_Нейтральная КИ'), (23, 'OKB_RATING_SCORING_Ошибка выполнения'), (24, 'OKB_RATING_SCORING_Плохая КИ'), (25, 'NBKI_RATING_SCORING_Хорошая КИ'), (26, 'NBKI_RATING_SCORING_Нейтральная КИ'), (27, 'NBKI_RATING_SCORING_КИ отсутствует'), (28, 'NBKI_RATING_SCORING_Ошибка выполнения'), (29, 'EQUIFAX_RATING_SCORING_КИ отсутствует'), (30, 'EQUIFAX_RATING_SCORING_Хорошая КИ'), (31, 'EQUIFAX_RATING_SCORING_Нейтральная КИ'), (32, 'EQUIFAX_RATING_SCORING_Ошибка выполнения')]
        new_cols = X.columns.tolist()
        # for i, c in enumerate(orig_cols):
        #     print(c[1], new_cols[i], c[1] == new_cols[i])
        # exit()
        # assert [(i, v) for i, v in enumerate(X.columns)] == orig_cols
        # print(pd.DataFrame([(i, v) for i, v in enumerate(X.columns)]))
        X.columns = [i for i, v in enumerate(X.columns)]
        # print(X)
        # exit()
        for i, r in enumerate(X.itertuples(index=False, name=None)):
        # for g in X.iterrows():
        #     i, r = g
            res = []
            # -- 1
            if r[20]:  # 'OKB_RATING_SCORING_КИ отсутствует'
                res.append(self.v1a)  # higher - higher precision
            elif r[21] == 1:  # 'OKB_RATING_SCORING_Хорошая КИ'
                res.append(self.v1b)  # lower higher precision
            else:
                res.append(self.v1c)

            # -- 2 a b c
            res.append(r[6] * self.v2a - self.v2b)  # 'EQUIFAX_SCORING'

            # -- 3 a b c
            # low precision
            # x = r['CLIENT_WI_EXPERIENCE']
            # a1, b1 = 0.11162212531024945, 0.02
            # a2, b2 = 0.2232442506204989, 0.02
            # if r['OKB_RATING_SCORING_КИ отсутствует'] == 0:
            #     y = a1 * n~p.log(b1 * x)
            # else:
            #     y = a2 * np.log(b2 * x + 0.2)
            # res +=y
            # low recall
            if r[3] < self.v3a:  # 'CLIENT_WI_EXPERIENCE'
                res.append(self.v3b)  # higher - higher precision
            else:
                res.append(self.v3c)  # lower - higher precision
            # -- 4 a b
            # {0: 'в браке', 1: 'не в браке'}
            if r[1] == 1:  # 'CLIENT_MARITAL_STATUS'
                res.append(self.v4a)
            else:
                res.append(self.v4b)
            # -- 5 a b c
            if 0 <= r[8] < 50000:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5a)
            elif 50000 <= r[8] < 300000:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5b)
            elif 300000 > r[8]:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5c)
            # -- 6
            if r[7] <= 700:  # 'NBKI_SCORING'
                res.append(self.v6c)
            else:
                res.append(r[7] * self.v6a - self.v6b)  # 'NBKI_SCORING'
            # -- 7
            if r[20]:  # 'OKB_RATING_SCORING_КИ отсутствует'
                multiplier = self.v7a * r[20]  # 'OKB_RATING_SCORING_КИ отсутствует'
            else:
                multiplier = self.v7b * r[20]  # 'OKB_RATING_SCORING_КИ отсутствует'
            # if r['МБКИ_требаналотч'] == 0:
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
            #     res.append(self.v71b + self.v71b * multiplier)
            #
            # if r['МБКИ_треб_исп_пр'] == 0:
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
            #     res.append(self.v72b + self.v72b * multiplier)
            #
            # if r['МБКИ_нет_огр'] == 0:
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
            #     res.append(self.v73b + self.v73b * multiplier)
            if r[12] == 1:  # 'МБКИ_требаналотч'
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
                res.append(self.v71b * multiplier)

            if r[10] == 1:  # 'МБКИ_треб_исп_пр'
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
                res.append(self.v72b * multiplier)

            if r[13] == 1:  # 'МБКИ_нет_огр'
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
                res.append(self.v73b * multiplier)

            preds[i][0] = - sum([x for x in res if x <= 0])
            preds[i][1] = sum([x for x in res if x > 0])
        return preds




# -------------------------- WHITE ZONE ----------------
class MyEstimXGB_wz(BaseEstimator):
    def __init__(self, v1a=0, v1b=0, v1c=0, v2a=0, v2b=0,
                 v3a=0, v3b=0, v3c=0, v4a=0, v4b=0, v5a=0, v5b=0, v5c=0, v6a=0, v6b=0, v6c=0,
                 v7a=0, v7b=0, v71a=0, v71b=0, v72a=0, v72b=0, v73a=0, v73b=0):
        self.v1a = v1a
        self.v1b = v1b
        self.v1c = v1c
        self.v2a = v2a
        self.v2b = v2b
        self.v3a = v3a
        self.v3b = v3b
        self.v3c = v3c
        self.v4a = v4a
        self.v4b = v4b
        self.v5a = v5a
        self.v5b = v5b
        self.v5c = v5c
        self.v6a = v6a  # NBKI scoring
        self.v6b = v6b
        self.v6c = v6c
        self.v7a = v7a  # MBKI scoring factors # multiplier
        self.v7b = v7b  # MBKI scoring factors # multiplier
        # self.v71a = v71a
        self.v71b = v71b
        # self.v72a = v72a
        self.v72b = v72b
        # self.v73a = v73a
        self.v73b = v73b
        # for sklearn
        self.classes_ = 2

    def fit(self, X=None, y=None):
        self.classes_ = unique_labels(y)
        pass

    def predict(self, X):
        # return np.argmax(self.pred(X), axis=1)
        # print((self.pred(X)[:,1]>0).astype(int))
        # exit()
        return (self.pred(X)[:, 0] <= 0).astype(int)

    def score(self, X, y):
        # y_pred = np.argmax(self.pred(X), axis=1)
        # y_pred = self.pred(X)[:, 1]
        y_pred = (self.pred(X)[:, 0] <= 0).astype(int)
        return metrics.accuracy_score(y, y_pred)

    def predict_proba(self, X):  # something wrong

        # z = np.array(list(zip(np.zeros(len(X))+0.1, self.pred(X))))
        # print(z[0], z[1])
        # z = np.argmax(, axis=1)
        return self.pred(X)

    def get_params(self, deep=True):
        return {
            "v1a": self.v1a,
            "v1b": self.v1b,
            "v1c": self.v1c,
            "v2a": self.v2a,
            "v2b": self.v2b,
            "v3a": self.v3a,
            "v3b": self.v3b,
            "v3c": self.v3c,
            "v4a": self.v4a,
            "v4b": self.v4b,
            "v5a": self.v5a,
            "v5b": self.v5b,
            "v5c": self.v5c,
            "v6a": self.v6a,
            "v6b": self.v6b,
            "v6c": self.v6c,
            "v7a": self.v7a,
            "v7b": self.v7b,
            # "v71a": self.v71a,
            "v71b": self.v71b,
            # "v72a": self.v72a,
            "v72b": self.v72b,
            # "v73a": self.v73a,
            "v73b": self.v73b
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # self.params_est = parameters
        return self

    # for parameter, value in parameters.items():

    # def decision_function(self, X):
    #     z = np.array(list(zip(np.zeros(len(X)), self.pred(X))))
    #     print(z[0], z[1])
    #     return z

    def pred(self, X: pd.DataFrame):
        preds = np.zeros(shape=(X.shape[0], 2), dtype=np.float)
        X = X.reset_index(drop=True).copy()
        # for i, r in X.iterrows():
        from transliterate import translit
        # X.columns = [translit(v, 'ru', reversed=True) for v in X.columns]
        # -- UNCOMMENT TO ENSURE RIGHT ORDER
        # print([(i, v) for i, v in enumerate(X.columns)])
        orig_cols = [(0, 'CLIENT_AGE'), (1, 'CLIENT_MARITAL_STATUS'), (2, 'CLIENT_DEPENDENTS_COUNT'), (3, 'CLIENT_WI_EXPERIENCE'), (4, 'ANKETA_SCORING'), (5, 'OKB_SCORING'), (6, 'EQUIFAX_SCORING'), (7, 'NBKI_SCORING'), (8, 'AUTO_DEAL_INITIAL_FEE'), (9, 'МБКИ_треб_адрес'), (10, 'МБКИ_треб_исп_пр'), (11, 'МБКИ_треб_пассп'), (12, 'МБКИ_требаналотч'), (13, 'МБКИ_нет_огр'), (14, 'МБКИ_недост'), (15, 'МБКИ_розыск'), (16, 'МБКИ_невыполнена'), (17, 'МБКИ_налспецуч'), (18, 'МБКИ_данные_не'), (19, 'День недели'), (20, 'OKB_RATING_SCORING_КИ отсутствует'), (21, 'OKB_RATING_SCORING_Хорошая КИ'), (22, 'OKB_RATING_SCORING_Нейтральная КИ'), (23, 'OKB_RATING_SCORING_Ошибка выполнения'), (24, 'OKB_RATING_SCORING_Плохая КИ'), (25, 'NBKI_RATING_SCORING_Хорошая КИ'), (26, 'NBKI_RATING_SCORING_Нейтральная КИ'), (27, 'NBKI_RATING_SCORING_КИ отсутствует'), (28, 'NBKI_RATING_SCORING_Ошибка выполнения'), (29, 'EQUIFAX_RATING_SCORING_КИ отсутствует'), (30, 'EQUIFAX_RATING_SCORING_Хорошая КИ'), (31, 'EQUIFAX_RATING_SCORING_Нейтральная КИ'), (32, 'EQUIFAX_RATING_SCORING_Ошибка выполнения')]
        new_cols = X.columns.tolist()
        # for i, c in enumerate(orig_cols):
        #     print(c[1], new_cols[i], c[1] == new_cols[i])
        # exit()
        # assert [(i, v) for i, v in enumerate(X.columns)] == orig_cols
        # print(pd.DataFrame([(i, v) for i, v in enumerate(X.columns)]))
        X.columns = [i for i, v in enumerate(X.columns)]
        # print(X)
        # exit()
        for i, r in enumerate(X.itertuples(index=False, name=None)):
        # for g in X.iterrows():
        #     i, r = g
            res = []
            # -- 1
            if r[20]:  # 'OKB_RATING_SCORING_КИ отсутствует'
                res.append(self.v1a)  # higher - higher precision
            elif r[21] == 1:  # 'OKB_RATING_SCORING_Хорошая КИ'
                res.append(self.v1b)  # lower higher precision
            else:
                res.append(self.v1c)

            # -- 2 a b c
            res.append(r[6] * self.v2a - self.v2b)  # 'EQUIFAX_SCORING'

            # -- 3 a b c
            # low precision
            # x = r['CLIENT_WI_EXPERIENCE']
            # a1, b1 = 0.11162212531024945, 0.02
            # a2, b2 = 0.2232442506204989, 0.02
            # if r['OKB_RATING_SCORING_КИ отсутствует'] == 0:
            #     y = a1 * n~p.log(b1 * x)
            # else:
            #     y = a2 * np.log(b2 * x + 0.2)
            # res +=y
            # low recall
            if r[3] < self.v3a:  # 'CLIENT_WI_EXPERIENCE'
                res.append(self.v3b)  # higher - higher precision
            else:
                res.append(self.v3c)  # lower - higher precision
            # -- 4 a b
            # {0: 'в браке', 1: 'не в браке'}
            if r[1] == 1:  # 'CLIENT_MARITAL_STATUS'
                res.append(self.v4a)
            else:
                res.append(self.v4b)
            # -- 5 a b c
            if 0 <= r[8] < 50000:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5a)
            elif 50000 <= r[8] < 300000:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5b)
            elif 300000 > r[8]:  # 'AUTO_DEAL_INITIAL_FEE'
                res.append(self.v5c)
            # -- 6
            if r[7] <= 700:  # 'NBKI_SCORING'
                res.append(self.v6c)
            else:
                res.append(r[7] * self.v6a - self.v6b)  # 'NBKI_SCORING'
            # -- 7
            if r[20]:  # 'OKB_RATING_SCORING_КИ отсутствует'
                multiplier = self.v7a * r[20]  # 'OKB_RATING_SCORING_КИ отсутствует'
            else:
                multiplier = self.v7b * r[20]  # 'OKB_RATING_SCORING_КИ отсутствует'
            # if r['МБКИ_требаналотч'] == 0:
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
            #     res.append(self.v71b + self.v71b * multiplier)
            #
            # if r['МБКИ_треб_исп_пр'] == 0:
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
            #     res.append(self.v72b + self.v72b * multiplier)
            #
            # if r['МБКИ_нет_огр'] == 0:
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
            #     res.append(self.v73b + self.v73b * multiplier)
            if r[12] == 1:  # 'МБКИ_требаналотч'
            #     res.append(self.v71a + self.v71a * multiplier)
            # else:  # == 1
                res.append(self.v71b * multiplier)

            if r[10] == 1:  # 'МБКИ_треб_исп_пр'
            #     res.append(self.v72a + self.v72a * multiplier)
            # else:  # == 1
                res.append(self.v72b * multiplier)

            if r[13] == 1:  # 'МБКИ_нет_огр'
            #     res.append(self.v73a + self.v73a * multiplier)
            # else:  # == 1
                res.append(self.v73b * multiplier)

            preds[i][0] = - sum([x for x in res if x <= 0])
            preds[i][1] = sum([x for x in res if x >= 0])
        return preds


# Accuracy: 0.702455
# AUC: 0.607324
# Precision: 0.714162
# Recall: 0.009383
# Одобренных: 0.003964
# MyEstimXGB(v1a=-0.325, v1b=0.2, v1c=0.15, v2a=0.6666666666666666, v2b=260,
#            v2c=2.5555555555555554, v3b=-0.38888888888888884, v3c=0.0, v4a=-0.2,
#            v4b=0.0)

# Accuracy: 0.709739
# AUC: 0.615533
# Precision: 0.565993
# Recall: 0.128828
# Одобренных: 0.068155
# MyEstimXGB(v1a=-0.325, v1b=0.2, v1c=0.22499999999999998, v2a=0.5555555555555556,
#            v2b=240, v2c=2.111111111111111, v3b=-0.31111111111111106, v3c=0.0,
#            v4a=-0.30000000000000004, v4b=0.22499999999999998)


# Accuracy: 0.733792
# AUC: 0.627347
# Precision: 0.523555
# Recall: 0.044711
# Одобренных: 0.022064
# MyEstimXGB(v1a=-0.2, v1b=0.2, v1c=0.0, v2a=0.5555555555555556, v2b=270,
#            v2c=2.333333333333333, v3a=70, v3b=-0.15555555555555545, v3c=0.25,
#            v4a=0.0, v4b=0.15)

# Accuracy: 0.734021
# AUC: 0.588845
# Precision: 0.523227
# Recall: 0.057257
# Одобренных: 0.028563
#
# MyEstimXGB(v1a=-0.44999999999999996, v1b=0.0, v1c=0.0, v2a=0.5555555555555556,
#            v2b=260, v2c=2.333333333333333, v3a=70, v3b=-0.15555555555555545,
#            v3c=0.5, v4a=-0.2, v4b=0.075)
# Accuracy: 0.733685
# AUC: 0.612087
# Precision: 0.527949
# Recall: 0.043969
# Одобренных: 0.021774
#
# MyEstimXGB(v1a=-0.325, v1c=0.15, v2a=0.5555555555555556, v2b=230,
#            v2c=2.5555555555555554, v3a=70, v3b=-0.31111111111111106, v3c=0.375)
# Accuracy: 0.735398
# AUC: 0.619688
# Precision: 0.552308
# Recall: 0.060736
# Одобренных: 0.029052
#
# MyEstimXGB(v1a=-0.4, v1c=0.15, v2a=0.55, v2b=245, v3a=70, v3b=-0.3, v3c=0.375,
#            v4a=-0.19999999999999998, v4b=-0.14444444444444446,
#            v5a=0.4222222222222222, v5b=-0.09999999999999998,
#            v5c=0.4222222222222222)
# (r['EQUIFAX_SCORING'] * 0.55) / 245 - 2.2)

# Accuracy: 0.735398
# AUC: 0.615522
# Precision: 0.550294
# Recall: 0.062390
# Одобренных: 0.029939
#
# MyEstimXGB(v1a=-0.325, v1c=0.15, v2a=0.55, v2b=245, v2c=2.25, v3a=70, v3b=-0.31,
#            v3c=0.375, v4a=-0.15, v4b=0.0, v5a=0.24285714285714285, v5b=-0.2,
#            v5c=0.26)
# {'v1a': -0.325, 'v1b': 0.1, 'v1c': 0.15, 'v2a': 0.55, 'v2b': 245, 'v2c': 2.25, 'v3a': 70, 'v3b': -0.31, 'v3c': 0.375, 'v4a': -0.15, 'v4b': 0.0, 'v5a': 0.24285714285714285, 'v5b': -0.2, 'v5c': 0.26}

# without
# Accuracy: 0.717424
# AUC: 0.659558
# Precision: 0.608094
# Recall: 0.235921
# Одобренных: 0.120094
#
# MyEstimXGB(v1a=-0.35, v1b=0.125, v1c=0.0, v2a=0.6000000000000001, v2b=220.0,
#            v3a=70.0, v3b=-0.1, v3c=0.2, v4a=-0.2, v4b=0.0, v5a=0.3, v5b=-0.125,
#            v5c=0.30000000000000004)
# {'v1a': -0.35, 'v1b': 0.125, 'v1c': 0.0, 'v2a': 0.6000000000000001, 'v2b': 220.0, 'v2c': 2.2, 'v3a': 70.0, 'v3b': -0.1, 'v3c': 0.2, 'v4a': -0.2, 'v4b': 0.0, 'v5a': 0.3, 'v5b': -0.125, 'v5c': 0.30000000000000004}


# Accuracy: 0.717660
# AUC: 0.669732
# Precision: 0.611788
# Recall: 0.231669
# Одобренных: 0.117242
#
# MyEstimXGB(v1a=-0.325, v1b=0.2, v1c=0.0, v2a=0.65, v2b=240.0, v2c=2.3,
#            v3a=63.75, v3b=-0.05, v3c=0.2, v4a=-0.25, v4b=0.1, v5a=0.25,
#            v5b=-0.2, v5c=0.36)
# {'v1a': -0.325, 'v1b': 0.2, 'v1c': 0.0, 'v2a': 0.65, 'v2b': 240.0, 'v2c': 2.3, 'v3a': 63.75, 'v3b': -0.05, 'v3c': 0.2, 'v4a': -0.25, 'v4b': 0.1, 'v5a': 0.25, 'v5b': -0.2, 'v5c': 0.36}

# mean_squared_error -0.29094517208999177
# Accuracy: 0.709054
# AUC: 0.670774
# Precision: 0.671678
# Recall: 0.123560
# Одобренных: 0.057177
# MyEstimXGB(v1a=-0.32, v1b=0.175, v1c=0.02500000000000001, v2a=0.002625,
#            v2b=2.35, v3a=62.5, v3b=-0.045, v3c=0.2, v4a=-0.3, v4b=-0.05,
#            v5a=0.225, v5b=-0.175, v5c=0.36, v6a=0.00116, v6b=0.828, v6c=-0.116,
#            v7=0.625, v71a=0.07500000000000001, v71b=-0.1, v72a=0.1, v72b=-0.05,
#            v73a=0.0, v73b=-0.1)
# {'v1a': -0.32, 'v1b': 0.175, 'v1c': 0.02500000000000001, 'v2a': 0.002625, 'v2b': 2.35, 'v3a': 62.5, 'v3b': -0.045, 'v3c': 0.2, 'v4a': -0.3, 'v4b': -0.05, 'v5a': 0.225, 'v5b': -0.175, 'v5c': 0.36, 'v6a': 0.00116, 'v6b': 0.828, 'v6c': -0.116, 'v7': 0.625, 'v71a': 0.07500000000000001, 'v71b': -0.1, 'v72a': 0.1, 'v72b': -0.05, 'v73a': 0.0, 'v73b': -0.1}


# mean_squared_error -0.28717450180497534
# Accuracy: 0.712825
# AUC: 0.664270
# Precision: 0.610272
# Recall: 0.214974
# Одобренных: 0.110173
# MyEstimXGB(v1a=-0.32, v1b=0.18, v1c=0.02, v2a=0.0026333333333333334, v2b=2.4,
#            v3a=61.666666666666664, v3b=-0.04, v3c=0.18333333333333332,
#            v4a=-0.35, v4b=1.3877787807814457e-17, v5a=0.21666666666666667,
#            v5b=-0.16333333333333333, v5c=0.35333333333333333,
#            v6a=0.0010999999999999998, v6b=0.835, v6c=-0.1225, v7=0.4, v71a=0.1,
#            v71b=-0.075, v72a=0.2, v72b=-0.03, v73a=-0.01, v73b=0.05)
# {'v1a': -0.32, 'v1b': 0.18, 'v1c': 0.02, 'v2a': 0.0026333333333333334, 'v2b': 2.4, 'v3a': 61.666666666666664, 'v3b': -0.04, 'v3c': 0.18333333333333332, 'v4a': -0.35, 'v4b': 1.3877787807814457e-17, 'v5a': 0.21666666666666667, 'v5b': -0.16333333333333333, 'v5c': 0.35333333333333333, 'v6a': 0.0010999999999999998, 'v6b': 0.835, 'v6c': -0.1225, 'v7': 0.4, 'v71a': 0.1, 'v71b': -0.075, 'v72a': 0.2, 'v72b': -0.03, 'v73a': -0.01, 'v73b': 0.05}
# mean_squared_error -0.3097173227235389
# Accuracy: 0.690282
# AUC: 0.642497
# Precision: 0.894771
# Recall: 0.003102
# Одобренных: 0.001148
# MyEstimXGB(v1a=-0.32, v1b=0.16, v1c=0.04, v2a=0.0023, v2b=2.35,
#            v3a=61.666666666666664, v3b=-0.04, v3c=0.15, v4a=-0.25, v4b=-0.1,
#            v5a=0.2, v5b=-0.17666666666666667, v5c=0.35333333333333333,
#            v6a=0.001, v6b=0.835, v6c=-0.1225, v7=0.8,
#            v71b=-0.012499999999999997, v72b=-0.08, v73b=-0.012499999999999997)
# {'v1a': -0.32, 'v1b': 0.16, 'v1c': 0.04, 'v2a': 0.0023, 'v2b': 2.35, 'v3a': 61.666666666666664, 'v3b': -0.04, 'v3c': 0.15, 'v4a': -0.25, 'v4b': -0.1, 'v5a': 0.2, 'v5b': -0.17666666666666667, 'v5c': 0.35333333333333333, 'v6a': 0.001, 'v6b': 0.835, 'v6c': -0.1225, 'v7': 0.8, 'v71a': 0, 'v71b': -0.012499999999999997, 'v72a': 0, 'v72b': -0.08, 'v73a': 0, 'v73b': -0.012499999999999997}

# mean_squared_error -0.288281038184173
# Accuracy: 0.711718
# AUC: 0.668401
# Precision: 0.648874
# Recall: 0.159334
# Одобренных: 0.076728
# MyEstimXGB(v1a=-0.32, v1b=0.19, v1c=0.02, v2a=0.0028, v2b=2.4,
#            v3a=61.666666666666664, v3b=-0.03, v3c=0.21666666666666667,
#            v4a=-0.2833333333333333, v4b=0.05, v5a=0.21666666666666667,
#            v5b=-0.19, v5c=0.3666666666666667, v6a=0.00105, v6b=0.83, v6c=-0.1,
#            v7=1.0, v71b=-0.2, v72b=-0.055, v73b=0.05)
# {'v1a': -0.32, 'v1b': 0.19, 'v1c': 0.02, 'v2a': 0.0028, 'v2b': 2.4, 'v3a': 61.666666666666664, 'v3b': -0.03, 'v3c': 0.21666666666666667, 'v4a': -0.2833333333333333, 'v4b': 0.05, 'v5a': 0.21666666666666667, 'v5b': -0.19, 'v5c': 0.3666666666666667, 'v6a': 0.00105, 'v6b': 0.83, 'v6c': -0.1, 'v7': 1.0, 'v71a': 0, 'v71b': -0.2, 'v72a': 0, 'v72b': -0.055, 'v73a': 0, 'v73b': 0.05}
# Accuracy: 0.710591
# AUC: 0.657421
# Precision: 0.661008
# Recall: 0.142503
# Одобренных: 0.067403
# MyEstimXGB(v1a=-0.32, v1b=0.175, v1c=0.02333333333333333,
#            v2a=0.0028666666666666667, v2b=2.466666666666667, v3a=61.0,
#            v3b=-0.02, v3c=0.23, v4a=-0.26, v4b=0.03266666666666667,
#            v5a=0.21333333333333332, v5b=-0.2, v5c=0.36,
#            v6a=0.0010333333333333334, v6b=0.84, v6c=-0.06666666666666667,
#            v7=0.9, v71b=-0.18, v72b=-0.07, v73b=0.1)
# {'v1a': -0.32, 'v1b': 0.175, 'v1c': 0.02333333333333333, 'v2a': 0.0028666666666666667, 'v2b': 2.466666666666667, 'v3a': 61.0, 'v3b': -0.02, 'v3c': 0.23, 'v4a': -0.26, 'v4b': 0.03266666666666667, 'v5a': 0.21333333333333332, 'v5b': -0.2, 'v5c': 0.36, 'v6a': 0.0010333333333333334, 'v6b': 0.84, 'v6c': -0.06666666666666667, 'v7': 0.9, 'v71b': -0.18, 'v72b': -0.07, 'v73b': 0.1}

# mean_squared_error -0.28815806846643455
# Accuracy: 0.711841
# AUC: 0.668235
# Precision: 0.646083
# Recall: 0.163294
# Одобренных: 0.079064

# MyEstimXGB(v1a=-0.32, v1b=0.175, v1c=0.01, v2a=0.0028666666666666667, v2b=2.5,
#            v3a=63.0, v3b=-0.03333333333333333, v3c=0.22, v4a=-0.26, v4b=0.1,
#            v5a=0.205, v5b=-0.19333333333333333, v5c=0.35,
#            v6a=0.0010666666666666667, v6b=0.8266666666666667,
#            v6c=-0.06666666666666667, v71b=-0.22, v72b=-0.04, v73b=0.1,
#            v7a=0.9777777777777776, v7b=0.0888888888888888)
# {'v1a': -0.32, 'v1b': 0.175, 'v1c': 0.01, 'v2a': 0.0028666666666666667, 'v2b': 2.5, 'v3a': 63.0, 'v3b': -0.03333333333333333, 'v3c': 0.22, 'v4a': -0.26, 'v4b': 0.1, 'v5a': 0.205, 'v5b': -0.19333333333333333, 'v5c': 0.35, 'v6a': 0.0010666666666666667, 'v6b': 0.8266666666666667, 'v6c': -0.06666666666666667, 'v7a': 0.9777777777777776, 'v7b': 0.0888888888888888, 'v71b': -0.22, 'v72b': -0.04, 'v73b': 0.1}

# mean_squared_error -0.2889778287885584
# Accuracy: 0.711022
# AUC: 0.667082
# Precision: 0.659556
# Recall: 0.146925
# Одобренных: 0.069719
# MyEstimXGB(v1a=-0.32, v1b=0.17, v1c=0.05, v2a=0.002766666666666667, v2b=2.4,
#            v3a=62.666666666666664, v3b=-0.02, v3c=0.2, v4a=-0.27,
#            v4b=0.05333333333333333, v5a=0.20333333333333334, v5b=-0.195,
#            v5c=0.34, v6a=0.0010666666666666667, v6b=0.84, v6c=-0.05, v71b=-0.26,
#            v72b=-0.039999999999999994, v73b=0.15, v7a=0.8888888888888891,
#            v7b=0.0)
# {'v1a': -0.32, 'v1b': 0.17, 'v1c': 0.05, 'v2a': 0.002766666666666667, 'v2b': 2.4, 'v3a': 62.666666666666664, 'v3b': -0.02, 'v3c': 0.2, 'v4a': -0.27, 'v4b': 0.05333333333333333, 'v5a': 0.20333333333333334, 'v5b': -0.195, 'v5c': 0.34, 'v6a': 0.0010666666666666667, 'v6b': 0.84, 'v6c': -0.05, 'v7a': 0.8888888888888891, 'v7b': 0.0, 'v71b': -0.26, 'v72b': -0.039999999999999994, 'v73b': 0.15}


# mean_squared_error -0.3026395776675732
# Accuracy: 0.697360
# AUC: 0.680573
# Precision: 0.696169
# Recall: 0.156667
# Одобренных: 0.075252
# Просрочка больше 1 дня 3
# Непрерывная Просрочка больше 5 дней 2
# MyEstimXGB(v1a=-0.32, v1b=0.17, v1c=0.04, v2a=0.0027, v2b=2.27, v3a=63.0,
#            v3b=-0.030000000000000002, v3c=0.2, v4a=-0.27749999999999997,
#            v4b=0.048, v5a=0.205, v5b=-0.2, v5c=0.355, v6a=0.001, v6b=0.815,
#            v6c=-0.38333333333333336, v71b=-0.23666666666666666,
#            v72b=-0.05633333333333333, v73b=0.14300000000000002, v7a=0.93,
#            v7b=0.21333333333333335)
# {'v1a': -0.32, 'v1b': 0.17, 'v1c': 0.04, 'v2a': 0.0027, 'v2b': 2.27, 'v3a': 63.0, 'v3b': -0.030000000000000002, 'v3c': 0.2, 'v4a': -0.27749999999999997, 'v4b': 0.048, 'v5a': 0.205, 'v5b': -0.2, 'v5c': 0.355, 'v6a': 0.001, 'v6b': 0.815, 'v6c': -0.38333333333333336, 'v7a': 0.93, 'v7b': 0.21333333333333335, 'v71b': -0.23666666666666666, 'v72b': -0.05633333333333333, 'v73b': 0.14300000000000002}

# mean_squared_error -0.3007518796992481
# Accuracy: 0.699248
# AUC: 0.681695
# Precision: 0.694301
# Recall: 0.168116
# Одобренных: 0.080947
# Просрочка больше 1 дня 6
# Непрерывная Просрочка больше 5 дней 2
# MyEstimXGB(v1a=-0.32, v1b=0.1698, v1c=0.04, v2a=0.00271, v2b=2.269, v3a=62.5,
#            v3b=-0.04, v3c=0.19, v4a=-0.278, v4b=0.051, v5a=0.2, v5b=-0.2,
#            v5c=0.362, v6a=0.001, v6b=0.81, v6c=-0.28, v71b=-0.26, v72b=-0.053,
#            v73b=0.15, v7a=0.92, v7b=0.18)
# {'v1a': -0.32, 'v1b': 0.1698, 'v1c': 0.04, 'v2a': 0.00271, 'v2b': 2.269, 'v3a': 62.5, 'v3b': -0.04, 'v3c': 0.19, 'v4a': -0.278, 'v4b': 0.051, 'v5a': 0.2, 'v5b': -0.2, 'v5c': 0.362, 'v6a': 0.001, 'v6b': 0.81, 'v6c': -0.28, 'v7a': 0.92, 'v7b': 0.18, 'v71b': -0.26, 'v72b': -0.053, 'v73b': 0.15}


