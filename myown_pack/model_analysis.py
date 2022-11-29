import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# own
from chepelev_pack.common import save, load
from chepelev_pack.plot import make_confusion_matrix


def check_model_sklearn_cross(est, X, y, splits=5, calc_confusion_matrix=True):
    kfold = StratifiedKFold(n_splits=splits)
    results = cross_val_score(est, X, y, cv=splits, scoring='neg_mean_squared_error')
    print("mean_squared_error", results.mean())
    results = cross_validate(est, X, y, cv=kfold, scoring=['accuracy', 'roc_auc', 'precision', 'recall'])
    # print(est.__class__.__name__)
    print("Accuracy: %f" % results['test_accuracy'].mean())
    print("AUC: %f" % results['test_roc_auc'].mean())
    print("Precision: %f" % results['test_precision'].mean())
    print("Recall: %f" % results['test_recall'].mean())

    # -- confustion matrix
    if calc_confusion_matrix:
        y_pred = cross_val_predict(est, X, y, cv=kfold)

        odob = sum(y_pred) / len(y)
        print("Одобренных: %f" % odob)
        print()

        print(classification_report(y, y_pred))
        print()

        cf_matrix = confusion_matrix(y, y_pred)

        # labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        # categories = ['Zero', 'One']
        labels = ['Верно отлоненных', 'Ошиб. одобренных', 'Ошиб. отклоненных', 'Верно одобренных']
        categories = ['отклонение', 'Одобрение']
        make_confusion_matrix(cf_matrix,
                              group_names=labels,
                              categories=categories,
                              cmap='binary')

        plt.savefig("confustuin matrix_cross")


def check_model_sklearn_split(*args, calc_confusion_matrix=True):
    """

    :param args: est, X_train, y_train, X_test, y_test  -- or -- est, X_test, y_test
    :return:
    """
    from sklearn import metrics

    if len(args) == 5:
        est, X_train, y_train, X_test, y_test = args
        est.fit(X_train, y_train)  # -- train
    else:
        est, X_test, y_test = args

    # test
    print(type(X_test), type(y_test), X_test.shape, y_test.shape)
    y_pred = est.predict(X_test)
    print("sum(y_pred)", sum(y_pred))
    print("len(y_pred) - sum(y_pred)", len(y_pred) - sum(y_pred))
    ac = metrics.accuracy_score(y_test, y_pred)
    pes = metrics.precision_score(y_test, y_pred)
    y_score = est.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_score)
    recall = metrics.recall_score(y_test, y_pred)
    odob = sum(y_pred)/len(y_test)
    print("Одобренных: %f" % odob)
    print("Accuracy: %f" % ac)
    print("AUC: %f" % auc)
    print("Precision: %f" % pes)
    print("Recall: %f" % recall)
    print()
    if calc_confusion_matrix:
        cf_matrix = confusion_matrix(y_test, y_pred)
        # labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        # categories = ['Zero', 'One']
        labels = ['Верно отлоненных', 'Ошиб. одобренных', 'Ошиб. отклоненных', 'Верно одобренных']
        categories = ['отклонение', 'Одобрение']
        make_confusion_matrix(cf_matrix,
                              group_names=labels,
                              categories=categories,
                              cmap='binary')

        plt.savefig("confustuin matrix_split")
    return ac, pes, recall


def forest_search_parameters(p: str, target: str, n_iter=20, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer
    from sklearn import metrics

    from sklearn.model_selection import train_test_split
    # -- data
    target = 'ander'
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    X: pd.DataFrame = df.drop([target], 1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
                                                        random_state=1, stratify=None)
    # -- metric
    def my_scoring_func(y_true, y_pred):
        p = metrics.precision_score(y_true, y_pred, zero_division=0)
        a = metrics.accuracy_score(y_true, y_pred)
        return (a * 2.5 + p) / 2

    score = make_scorer(my_scoring_func, greater_is_better=True)
    # -- search
    kfold = StratifiedKFold(n_splits=2)
    # RandomForestClassifier(class_weight={0: 5, 1: 1}, max_depth=8,
    #                        max_leaf_nodes=200, max_samples=0.5, n_estimators=180)
    # RandomForestClassifier(class_weight={0: 5, 1: 1}, max_depth=12,
    #                        max_leaf_nodes=140, n_estimators=290)
    # RandomForestClassifier(class_weight={0: 4, 1: 1}, max_depth=10,
    #                        max_features='sqrt', min_samples_split=10,
    #                        n_estimators=270)
    params = {'n_estimators': range(60, 140, 6),
              # 'max_leaf_nodes': range(50, 500, 10),
              'max_depth': list(range(2, 10, 1)),
              # 'class_weight': [{0:1.5,1:1}, {0:2,1:1}, {0:3,1:1}], #  {0:2,1:1}, {0:5,1:1} 'balanced', 'balanced_subsample',
              'max_features': ['sqrt'],
              # 'min_samples_leaf': [1],  # ,2,4
              # 'min_samples_split': [2],  # , 3, 5, 10
              # "min_samples_split": np.linspace(0.1, 0.5, 12),
              # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
              # 'max_samples':[None, 0.8, 0.9],
              # 'ccp_alpha':[0, 0.01, 0.1]
              'class_weight': [{0: 1, 1: 0.75}]
              }
    clf = RandomizedSearchCV(RandomForestClassifier(random_state=random_state), params, cv=kfold, scoring=score,# scoring='roc_auc', #
        n_jobs=4, verbose=1, n_iter=n_iter, random_state=random_state)

    parameters = {
        "loss": ["deviance"],
        "learning_rate": [0.1, 0.2,0.3],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        #"max_depth": [3, 22, 3],
        "max_features": ["log2", "sqrt"],
        #"criterion": ["friedman_mse"],
        #"subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators": range(50, 130, 5)
    }
    # from sklearn.ensemble import GradientBoostingClassifier
    # clf = RandomizedSearchCV(GradientBoostingClassifier(random_state=random_state), parameters, cv=kfold, n_jobs=4,
    #                          n_iter=n_iter, random_state=random_state, scoring=score)

    parameters = {
        # 'criterion': ['entropy'],  # 'gini',
        # "splitter": ["best", "random"],
        "splitter": ["best"],
        "max_depth": range(7, 40, 6),
        "min_samples_leaf": range(4, 12, 1),
        # "min_weight_fraction_leaf": [0],  #, 0.1, 0.2, 0.3, 0.4, 0.5
        # "max_features": ["auto", "log2", "sqrt", None],
        "max_leaf_nodes": list(range(20, 160, 2)),  # +[None],
        # 'class_weight': [{0: 2, 1: 1}, {0: 1.7, 1: 1}, {0: 1.3, 1: 1}, None]
        # 'class_weight': [{0: 2, 1: 1}, {0: 1.7, 1: 1}, {0: 1.3, 1: 1}, None]
        'class_weight': [{0: 1, 1: 1.35}]
        }

    # clf = RandomizedSearchCV(DecisionTreeClassifier(random_state=random_state), parameters, cv=kfold, n_jobs=4,
    #                          n_iter=n_iter, random_state=random_state, verbose=2) # scoring=score


    results = clf.fit(X, y)
    # print(results)
    print(results.best_estimator_)
    check_model_sklearn_cross(results.best_estimator_, X_train, y_train)


def search_parameters_own_model(model, p1: str, p2: str, target: str, n_iter=20, random_state=42):
    from sklearn.metrics import make_scorer
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from mart_procs import remove_special_cases
    from download import download_mart_oracle
    # -- data
    # -- data
    df_train: pd.DataFrame = pd.read_pickle(p1)
    df_test: pd.DataFrame = pd.read_pickle(p2)

    # df = remove_special_cases(df)
    time_ep = time.time()

    # -- orig
    p = 'by_hands.pickle'
    df_o = load(p)
    df_o.set_index('id', inplace=True)

    # -- mark prosrochka+7 as rej
    df_train_full: pd.DataFrame = df_o.loc[df_train["id"].to_list()]
    # l_rel = set(df_train_full[df_train_full['DEAL_STATUS'] == 'release']['DEAL_ID'])

    # l_rel = set(d['DEAL_ID'].tolist())  # train released
    # print("len(set(l_rel))", len(set(l_rel)))
    # p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
    #                          sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in ',
    #                          cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'],
    #                          l=list(l_rel))
    p = 'mart_cred_train_sel.pickle'
    df_cred: pd.DataFrame = pd.read_csv(p)

    # df_train.set_index('DEAL_ID', inplace=True)

    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 7)

    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()

    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)
    prosr_deal_id = df_cred[df_cred['pros_long_has'] > 0].index.tolist()
    print("len(deal_id_prosr)", len(prosr_deal_id))
    prosr_id = df_o[df_o['DEAL_ID'].isin(prosr_deal_id)].index
    print("len(prosr_id)", len(prosr_id))
    # exit()

    # -- double records with DEAL_STATUS="true"

    # d2 = df_train.set_index(keys=['id'], drop=True).loc[prosr_id].copy()
    # d2['ander'] = 0
    # df_train = pd.concat([df_train, d2, d2, d2], axis=0, sort=False)

    X_train: pd.DataFrame = df_train.drop([target, 'id'], 1)
    X_test: pd.DataFrame = df_test.drop([target, 'id'], 1)
    y_train = df_train[target]
    y_test = df_test[target]

    # -- score
    def my_scoring_func(y_true, y_pred):
        p = metrics.precision_score(y_true, y_pred)
        a = metrics.accuracy_score(y_true, y_pred)
        odob = sum(y_pred) / len(y_pred)
        if odob > 0.1 or 0.05 > odob:
            # print("odob bad", odob)
            return 0.01
        # else:
            # print(odob)
        r = a + p / 8
        # print(a, p, r, odob)
        return r
    score = make_scorer(my_scoring_func, greater_is_better=True)
    # -- model
    {'v1a': -0.32, 'v1b': 0.17, 'v1c': 0.04, 'v2a': 0.0027, 'v2b': 2.27, 'v3a': 63.0, 'v3b': -0.035, 'v3c': 0.2,
     'v4a': -0.2775, 'v4b': 0.05, 'v5a': 0.2, 'v5b': -0.2, 'v5c': 0.359, 'v6a': 0.001, 'v6b': 0.811, 'v6c': -0.3,
     'v7a': 0.91, 'v7b': 0.16666666666666669, 'v71b': -0.25, 'v72b': -0.055, 'v73b': 0.14500000000000002}
    # parameters = {
    #         "v1a": [-0.32],
    #         "v1b": np.linspace(0.1698, 0.1702, 3),
    #         "v1c": np.linspace(0.039, 0.041, 3),
    #         "v2a": np.linspace(0.00269, 0.00271, 3),
    #         "v2b": np.linspace(2.269, 2.271, 3),
    #         "v3a": np.linspace(62.5, 63.5, 3),
    #         "v3b": np.linspace(-0.03, -0.04, 6),
    #         "v3c": np.linspace(0.19, 0.21, 3),
    #         "v4a": np.linspace(-0.277, -0.279, 3),
    #         "v4b": np.linspace(0.049, 0.051, 3),
    #         "v5a": np.linspace(0.190, 0.21, 3),
    #         "v5b": np.linspace(-0.199, -0.201, 3),
    #         "v5c": np.linspace(0.350, 0.362, 7),
    #
    #         "v6a": np.linspace(0.00095, 0.001, 3),
    #         "v6b": np.linspace(0.810, 0.813, 3),
    #         "v6c": np.linspace(-0.28, -0.32, 4),
    #
    #         "v7a": np.linspace(0.90, 0.92, 3),
    #         "v7b": np.linspace(0.14, 0.18, 4),
    #         # "v71a": np.linspace(0.2, 0.1, 5),
    #         "v71b": np.linspace(-0.24, -0.26, 3),
    #         # "v72a": np.linspace(0.5, 0.2, 5),
    #         "v72b": np.linspace(-0.053, -0.057, 3),
    #         # "v73a": np.linspace(-0.01, 0.1, 5),
    #         "v73b": np.linspace(0.143, 0.157, 3),
    #     }
    # -- white zone
    parameters = {
        "v1a": [-1],
        "v1b": [+1],
        "v1c": [+1],
        "v2a": np.linspace(-0.00069, 0.00371, 10),
        "v2b": np.linspace(2.169, 2.471, 10),
        "v3a": [+1],
        "v3b": [+1],
        "v3c": [+1],
        "v4a": [+1],
        "v4b": [+1],
        "v5a": [+1],
        "v5b": [+1],
        "v5c": [-1],

        "v6a": np.linspace(-0.00005, 0.011, 10),
        "v6b": np.linspace(0.410, 0.913, 10),
        # "v6a": np.linspace(-0.00095, 0.001, 3),
        # "v6b": np.linspace(-0.810, 0.813, 3),
        "v6c": [+1],

        "v7a": [+1],
        "v7b": [-1],
        # "v71a": np.linspace(0.2, 0.1, 5),
        "v71b": [+1],
        # "v72a": np.linspace(0.5, 0.2, 5),
        "v72b": [-1],
        # "v73a": np.linspace(-0.01, 0.1, 5),
        "v73b": [-1],
    }

        # {
        # # 'criterion': ['entropy'],  # 'gini',
        # # "splitter": ["best", "random"],
        # "splitter": ["best"],
        # "max_depth": range(10, 70, 5),
        # "min_samples_leaf": range(1, 10, 1),
        # # "min_weight_fraction_leaf": [0],  #, 0.1, 0.2, 0.3, 0.4, 0.5
        # # "max_features": ["auto", "log2", "sqrt", None],
        # "max_leaf_nodes": list(range(30, 60, 2)),  # +[None],
        # "class_weight": [{0: 2, 1: 1}, {0: 1.7, 1: 1}, {0: 1.3, 1: 1}, None]}

    kfold = StratifiedKFold(n_splits=2)

    clf = RandomizedSearchCV(model, parameters, cv=kfold, n_jobs=4,
                             n_iter=n_iter, random_state=random_state, verbose=2, scoring=score)
    results = clf.fit(X_train, y_train)
    check_model_sklearn_cross(results.best_estimator_, X_train, y_train, splits=5)
    print(results.best_estimator_)
    print(results.best_estimator_.get_params())
    print("time", (time.time() - time_ep)/60/60, "hours")
    return results.best_estimator_


def xgb_search_parameters(p1: str, p2: str, target: str, n_iter=20, random_state=42, ignore: list = None):
    """ p1 for search parameters. p2 for test"""
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import make_scorer
    from sklearn import metrics
    # -- data

    df_train: pd.DataFrame = pd.read_pickle(p1)

    X_train: pd.DataFrame = df_train.drop(columns=[target, 'id'])
    y_train = df_train[target]

    # X = StandardScaler().fit_transform(X)  # XGB specific
    # # holdout for final check
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=random_state)
    kfold = StratifiedKFold(n_splits=2)

    # -- search
    # from sklearn.impute import SimpleImputer
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import OneHotEncoder
    # from sklearn.preprocessing import StandardScaler
    # categorical_pipeline = Pipeline(
    #     steps=[
    #         ("impite", SimpleImputer(strategy="most_frequent")),
    #         ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False))
    #     ]
    # )
    # numeric_pipeline = Pipeline(
    #     steps=[
    #         ("impute", SimpleImputer(strategy="mean")),
    #         ("scale", StandardScaler())
    #     ]
    # )
    # cat_cols = X.select_dtypes(exclude="number").columns
    # num_cols = X.select_dtypes(include="number").columns
    # print(cat_cols.shape)
    # print(num_cols.shape)
    # np.set_printoptions(threshold=np.inf)
    # print(np.round(np.cov(X.T),2))
    # return
    scale_pos_weight = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]
    print("scale_pos_weight", scale_pos_weight)

    # -- search
    estimator = XGBClassifier(
        booster='gbtree',  # main algorithm 'dart' - too slow, 'gblinear'
        objective='binary:logistic',
        nthread=4,
        seed=random_state,
        use_label_encoder=False,
        verbosity=1,
        scale_pos_weight=0.4,  # if we care for AUC
        gpu_id=0,
        eval_metric='logloss',
        random_state=random_state,
    )

    sc = scale_pos_weight
    parameters = {
        'learning_rate': [0.09, 0.05, 0.02],  # default = 0.3
        "gamma": [0.1, 0],  # 1) default 0 # 0, 0.25, 3
        'max_depth': range(2, 30, 2),  # 1) default 6
        # 'min_child_weight': range(3, 8, 1),  # 1) default 1
        'n_estimators': range(100, 150, 2),
        # "reg_lambda": [1],  # default=1
        "reg_alpha": [0.65, 0.58, 0.5],  # default = 0 "reg_alpha": [0.5, 0.2, 1],  # default = 0
        # "subsample": [0.8, 1],  # default = 1 (0,1)
        # "colsample_bytree": [0.5, 0.6, 1],  # default = 1 [0.5, 0.8, 1]
        # "scale_pos_weight": [sc-2.0, sc-1.0],   # if we care for AUC
        # "scale_pos_weight": [0.32],  # if we care for AUC
    }

    def my_scoring_func(y_true, y_pred):
        p = metrics.precision_score(y_true, y_pred)
        a = metrics.accuracy_score(y_true, y_pred)
        r = a*8 + p
        print('acc:', a, 'prec:',p, 'own:',r)
        return r
        # return (a * 2 + p) / 2

    score = make_scorer(my_scoring_func, greater_is_better=True)

    results = RandomizedSearchCV(
        estimator,
        parameters,
        scoring=score,
        # scoring='precision',
        # scoring='accuracy',
        n_jobs=4,
        cv=kfold,
        verbose=0,
        n_iter=n_iter
    )
    results.fit(X_train, y_train)
    print(results.best_estimator_)
    print(results.best_estimator_.get_params())

    check_model_sklearn_cross(results.best_estimator_, X_train, y_train)


def _permutation_importance(X: pd.DataFrame, y: pd.Series):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import RidgeCV

    X = StandardScaler().fit_transform(X)  # XGB specific
    # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                                     colsample_bynode=1, colsample_bytree=0.8, gamma=2, gpu_id=-1,
    #                                     importance_type='gain', interaction_constraints='',
    #                                     learning_rate=0.05, max_delta_step=0, max_depth=8,
    #                                     min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #                                     n_estimators=270, n_jobs=2, nthread=2, num_parallel_tree=1,
    #                                     random_state=42, reg_alpha=0.5, reg_lambda=0,
    #                                     scale_pos_weight=0.45948559272867917, seed=42, subsample=1,
    #                                     tree_method='exact', use_label_encoder=False,
    #                                     validate_parameters=1, verbosity=1)

    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
                      importance_type='gain', interaction_constraints='',
                      learning_rate=0.02, max_delta_step=0, max_depth=6,
                      min_child_weight=1, missing=np.nan, monotone_constraints='()',
                      n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
                      random_state=42, reg_alpha=0.5, reg_lambda=1,
                      seed=42, subsample=1, scale_pos_weight=0.28,  # 1.3346062351320898 # 0.36
                      tree_method='exact', use_label_encoder=False,
                      validate_parameters=1, verbosity=1)
    # model = XGBClassifier()
    model.fit(X, y)

    im1 = permutation_importance(model, X, y, n_repeats=5).importances_mean
    # --
    model = RidgeCV()
    model.fit(X, y)
    im2 = permutation_importance(model, X, y, n_repeats=1).importances_mean

    im1 = np.reshape(im1, [-1, 1])
    im1 = MinMaxScaler().fit_transform(im1)
    im1 = np.ravel(im1)
    im2 = np.reshape(im2, [-1, 1])
    im2 = MinMaxScaler().fit_transform(im2)
    im2 = np.ravel(im2)
    print(im1.shape)
    importance_sum = im1 + im2
    print(importance_sum.shape)
    # print(results)
    return importance_sum  # importance


def drop_collinear_columns(p, target='ander', lim_max: float = 0.98, drop=None, method:str='pearson'):
    """ and ander == 1 or 0 only"""
    import seaborn as sns
    from matplotlib import pyplot as plt
    df_orig: pd.DataFrame = pd.read_pickle(p)
    if drop:
        df_orig.drop(columns=drop, inplace=True)
    # print(df_orig.describe().to_string())
    # exit()
    # -- prepare data
    # df_orig = df_orig[(df_orig[target] == 0) | (df_orig[target] == 1)]
    X = df_orig.drop(columns=[target, 'id'])
    y = df_orig[target]
    df = X
    # -- calc correlation
    corr = df.dropna().corr()
    # print(corr.to_string())

    mask = np.zeros_like(corr, int)
    np.fill_diagonal(mask, 1)
    corr.iloc[mask.astype(bool)] = 0

    lim_max = lim_max

    df_too_much = None  # (> 0.98) for 1
    for c in corr.columns:
        if abs(corr[c].max()) > lim_max or abs(corr[c].min()) > lim_max:
            if df_too_much is None:
                df_too_much = pd.DataFrame(df[c])
            else:
                df_too_much[c] = df[c]

    # corr for selected columns
    if df_too_much is None:
        print("drop_collinear_columns: no correlation found")
        return save('drop_collinear_columns.pickle', df_orig)
    corr = df_too_much.corr(method=method)  # corr
    # fill diagonal 0
    mask = np.zeros_like(corr, int)
    np.fill_diagonal(mask, 1)
    corr.iloc[mask.astype(bool)] = 0

    # -- calc importance
    from sklearn.inspection import permutation_importance
    from sklearn import linear_model
    # est = linear_model.RidgeClassifierCV()
    w = _permutation_importance(X, y)  # importance
    cdf = pd.DataFrame(w, columns=['weight'])
    cdf['column'] = X.columns
    cdf_p = cdf.sort_values(by=['weight'], ascending=False)
    print("permutation importance:")
    print(cdf_p.to_string())
    print()
    cdf_p = cdf_p.set_index('column')
    # print(cdf_p.to_string(), '\n')
    # -- filter
    # with sns.axes_style("white"):
    #     f, ax = plt.subplots(figsize=(7, 7))
    #     ax = sns.heatmap(corr, square=True, linewidths=.8, cmap="YlGnBu")  # mask=mask,
    #     ax.set_title("Главная диаграмма корреляции")
    #     f.subplots_adjust(left=0.49, bottom=0.4)
    plt.show()
    print("before:", df_orig.shape)
    was = []
    for c in corr.columns:
        if c in was:
            continue
        max = corr[c].max()
        max_corr = corr[corr[c] == max].head(1)
        c_second = max_corr.index.values[0]
        if c_second in was:
            continue
        # print(c, c_second)

        c_prior = cdf_p.loc[c]['weight']
        # print(c, c_prior, cdf_p.loc[c])
        c_second_prior = cdf_p.loc[c_second]['weight']
        if c_prior > c_second_prior:
            df_orig.drop(c, axis=1, inplace=True)
            df_too_much.drop(c, axis=1, inplace=True)
            was.append(c)
        else:
            df_orig.drop(c_second, axis=1, inplace=True)
            df_too_much.drop(c_second, axis=1, inplace=True)
            was.append(c_second)

    # -- test
    # print(df_too_much.shape)
    print("deleted:")
    [print(c) for c in was]
    # if df_too_much.shape[1] != 0:
    #     corr = df_too_much.corr()  # corr
    #     with sns.axes_style("white"):
    #         f, ax = plt.subplots(figsize=(7, 7))
    #         ax = sns.heatmap(corr, square=True, linewidths=.8, cmap="YlGnBu")  # mask=mask,
    #         ax.set_title("Главная диаграмма корреляции")
    #         f.subplots_adjust(left=0.49, bottom=0.4)
    #     plt.show()
    print("after:", df_orig.shape)
    return save('drop_collinear_co`lumns.pickle', df_orig), was


def permutation_importance_forest(p: str, target: str, nrep=5):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import make_scorer
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop(columns=[target, 'id'])
    y = df[target]

    # --
    params = {'n_estimators': range(80, 120, 5), 'min_samples_split': [2],  # 'max_leaf_nodes': range(250, 450, 10),
              'max_depth': range(6, 12, 1),
              # 'class_weight': [{0: 2, 1: 1}]
              }

    # est = RandomForestClassifier(random_state=7, max_features='sqrt')

    # Accuracy: 0.716397
    # AUC: 0.711844
    # Precision: 0.586273
    # Recall: 0.178279
    # est = RandomForestClassifier(max_depth=6, # class_weight={0: 2, 1: 1},
    #                              max_features='sqrt', n_estimators=80, random_state=7)

    # Accuracy: 0.700683
    # Precision: 0.633721
    # Recall: 0.027311
    est = RandomForestClassifier( # class_weight={0: 1, 1: 0.70},  # max_depth=4,
                                 max_features='sqrt')   # random_state=25

    def my_scoring_func(y_true, y_pred):
        p = metrics.precision_score(y_true, y_pred)
        a = metrics.accuracy_score(y_true, y_pred)
        return (a * 2.5 + p) / 2

    score = make_scorer(my_scoring_func, greater_is_better=True)

    importance_sum = np.zeros(X.shape[1], dtype=float)
    c = 0
    for i in range(3, 5):
        for j in range(3, 5):
            # it is faster than cross validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True,
                                                                random_state=i)
            kfold = StratifiedKFold(n_splits=i, shuffle=True)
            gs = RandomizedSearchCV(est, params, cv=kfold, n_iter=20, n_jobs=4, random_state=i + j*3,
                                    # scoring=score # scoring='roc_auc',
                                    )
            results = gs.fit(X_train, y_train)
            model = results.best_estimator_
            print(model)
            # score
            y_score = model.predict_proba(X_test)[:, 1]
            # auc = metrics.roc_auc_score(y_test, y_score)
            sc = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            # pes = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred)

            print("Accuracy: %f" % sc)
            # print("AUC: %f" % auc)
            # print("Precision: %f" % pes)
            # print("Recall: %f" % recall)

            # FEATURE IMPORTANCE
            # img = model.feature_importances_  # feature importance
            # print(type(img))
            # print(img)
            # print()

            imp = permutation_importance(model, X, y, n_repeats=nrep)
            imp = imp.importances_mean

            # scale
            imp = np.reshape(imp, [-1, 1])
            imp = preprocessing.MinMaxScaler().fit_transform(imp)
            imp = np.ravel(imp)

            importance_sum += imp

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking forest:", importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importance_sum[indices[f]] / 100))
    return importance_sum, c
