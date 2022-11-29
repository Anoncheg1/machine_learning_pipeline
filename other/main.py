import pandas as pd
import numpy as np
# own
from imputation import impute


def print_num_obj_cols(df: pd.DataFrame):
    categorial_columns = df.select_dtypes(include=["object"]).columns
    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    print("numerical:")
    print(df[numerical_columns].describe().to_string())
    print("categorical:")
    print(df[categorial_columns].describe().to_string())
    for c in categorial_columns:
        print(c, df[c].unique())


def csv_file_read():
    p = '/home/u2/evseeva/Отчет по сделкам(Андеррайтер) (31).csv'
    df = pd.read_csv(p, index_col=0, low_memory=False)

    print(df.loc[df['Статус заявки'] != 'Заявка отменена']['Статус заявки'])
    df['Статус заявки'] = df['Статус заявки'].apply(lambda x: True if x == 'Заявка отменена' else False)


    # print(p.describe())
    # print(df.columns)
    # return
    # print(p['Решение по заявке'].unique())  # 'Отмена' nan 'Одобрено' 'reworked'
    # -- удаляем лишние строки
    # удаляем не дошедшие до андера
    idx = df.index[df['Сделка дошла до Андерайтреа'] == 0]
    df = df.drop(idx)
    df.drop(['Сделка дошла до Андерайтреа'], 1, inplace=True) # больше столбец не нужен
    # удаляем строки в которых возможно был сделан перерасчет
    # idx = df.index[(df['СФ андерайтера'] == 1) & (df['СФ системы'] == 0)]
    # print(len(idx))
    # df = df.drop(idx)
    # print(df[df['СФ андерайтера'] == 1])
    # return

    # Y = df['СФ андерайтера']
    # print(len(Y[Y == 1]))

    # replace https:// in
    http_dirt_cols = ['Оценка кредитной истории ОКБ',
                      'Оценка кредитной истории Эквифакс',
                      'Оценка кредитной истории НБКИ']
    # print(df['Оценка кредитной истории НБКИ'])
    for c in http_dirt_cols:
        df[c] = df[c].str.extract(r"^([\w ]*)[^\(]?")
    # print(df['Оценка кредитной истории НБКИ'])
    # print(df.isna().sum())

    # Удаляем лишние
    l_col = ['ИНН работодателя',
             'id клиента',
             'ФИО клиента',
             'Наименование работодателя',
             'Unnamed: 23',
             'Unnamed: 24',
             'Unnamed: 25']

    df.drop(l_col, axis=1, inplace=True)

    # -- Отмечаем цифровые столбцы
    df['Скоринговый балл НБКИ Digital Score'] = df['Скоринговый балл НБКИ Digital Score'].str.extract(r"^(\d*)")
    df['Скоринговый балл НБКИ FiCO'] = df['Скоринговый балл НБКИ FiCO'].str.extract(r"^(\d*)")
    df['Коды отказа'] = df['Коды отказа'].str.extract(r"^(\d*)")
    # print(pd.to_numeric(df.loc[df['СФ андерайтера'] == 0]['Коды отказа'], errors='coerce').astype('Int64').to_string())
    # return

    int_col = ['Скоринговый балл НБКИ Digital Score',
               'Скоринговый балл ОКБ, основной скоринг бюро',
               'Скоринговый балл НБКИ FiCO',
               'Эквифакс 4Score',
               'Анкетный скоринг',
               'Коды отказа'
               ]
    df['Дата рождения клиента'] = pd.to_numeric(2021 - pd.to_datetime(df['Дата рождения клиента']).dt.year).astype(
        'Int32')
    for col in int_col:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    df['ПВ, %'] = df['ПВ, %'].round().astype('Int64')

    # -- объединяем скоринговый бал
    # пороверяем средние
    # print(df['Скоринговый балл НБКИ Digital Score'].mean())
    # print(df['Скоринговый балл НБКИ FiCO'].mean())
    # import matplotlib.pyplot as plt
    # df.boxplot(column=['Скоринговый балл НБКИ Digital Score', 'Скоринговый балл НБКИ FiCO'])
    # plt.show()

    df['Скоринговый балл НБКИ общ'] = df['Скоринговый балл НБКИ Digital Score'].fillna(0) + df[
        'Скоринговый балл НБКИ FiCO'].fillna(0)
    df.drop('Скоринговый балл НБКИ Digital Score', 1, inplace=True)
    df.drop('Скоринговый балл НБКИ FiCO', 1, inplace=True)

    # categorial_columns = df.select_dtypes(include=["object"]).columns
    # numerical_columns = df.select_dtypes(exclude=["object"]).columns
    # print("categorical:")
    # print(df[categorial_columns].describe().to_string())
    # print("numerical:")
    # print(df[numerical_columns].describe().to_string())

    # -- сохраняем
    pd.to_pickle(df, 'after_read.pickle')
    print('ok')
    return 'after_read.pickle'

    # Y = df['СФ андерайтера']
    # print(Y[Y == 1].shape)
    # return

    # print(df2.describe())


# Норма - отчет по сделкам - автокредиты РНБ
# что такое 'СФ андерайтера' = сстоп фактор
# что такое Статус заявки - на каком этапе находится -
# Решение по заявки - Одобрена и отменена - Андерайтор (одобрена)
# связь с решением андерайтора?
# код системы - если цифра - одобрил андератором    буквы - на доработку
# где были ли просрочки - ЦФТ
# ID заявки = ID сделки ( ЦФТ?)

# I
# L = 1 I = 0 - перерасчет`


# скоринговый бал ОКБ и анкетный скоринг
# P R S U W
# s = P (переключился)

# статус договора

#

# def download():
#     select ldw.mart_norma_auto

def impute_v(p):
    df: pd.DataFrame = pd.read_pickle(p)
    df2 = impute(df)
    # correct na values
    df2['Коды отказа'] = df2['Коды отказа'].apply(lambda x: x if x >= 0 else 0)
    # df['Коды отказа'] = df['Коды отказа'].apply(lambda x: x if x >= 0 else 0)
    # -- сохраняем
    pd.to_pickle(df2, 'after_imputer.pickle')
    print('ok')
    return 'after_imputer.pickle'


def encode_categorical(p):
    """ check data and encode"""
    df: pd.DataFrame = pd.read_pickle(p)

    # print(df.loc[df['Статус заявки'] == 'Заявка отменена'])
    print(df['Статус заявки'])
    # return

    categorial_columns = df.select_dtypes(include=["object"]).columns
    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    print("categorical:")
    print(df[categorial_columns].head(30).to_string())
    print("numerical:")
    print(df[numerical_columns].head(30).to_string())
    # for c in categorial_columns:
    #     df[c] = df[c].astype('category')
    # print(df.head(30).to_string())
    # print(df['Оценка кредитной истории ОКБ'])
    # print(type(df['Оценка кредитной истории ОКБ'][103957]))
    for c in categorial_columns:
        print(c, len(df[df[c] == 'nan']))
    print(df.shape)

    # -- encode categorical One-Hot
    # df = pd.get_dummies(df, columns=categorial_columns, dummy_na=True)  # categorical
    # # for c in categorial_columns:
    # #     print(df[c].head().to_string())
    # print(df.head().to_string())
    # return

    # -- encode categorical Ordinary - better for feature importance
    label_encoders = {}
    from sklearn.preprocessing import LabelEncoder
    df.fillna(value='NoneMy', inplace=True)
    for c in categorial_columns:  # columns
        # save encoder
        le: LabelEncoder = LabelEncoder().fit(df[c])
        label_encoders[c] = le

        # index = np.where(le.classes_ == 'NoneMy')[0]
        # encode
        df[c] = pd.Series(le.transform(df[c]))
        # replace NoneMy to np.NaN
        # if index.shape[0] == 1 and c not in exception_columns:
        #     c_df[c].replace(index[0], np.nan, inplace=True)
    # print(df.head(30).to_string())
    # return

    # -- int32/int64 to int
    for c in df.columns:
        df[c] = df[c].astype(int)
    # df.apply(pd.to_numeric)

    # -- сохраняем
    p = 'encoded.pickle'
    pd.to_pickle(df, 'encoded.pickle')
    print('ok')
    return p


def feature_engeering(df):
    import featuretools as ft
    # df: pd.DataFrame = pd.read_pickle(p)

    df_target = df['target']
    df = df.drop(['target'], 1)

    es = ft.EntitySet()
    es.entity_from_dataframe(entity_id="df",
                             dataframe=df,
                             index="id")
    print(es)
    # trans_primitives_default = ["year", "month", "weekday", "haversine"]
    print(ft.primitives.list_primitives().to_string())
    # return

    trans_primitives = ['divide_numeric', 'add_numeric']
    # agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
    # primitive_options = {tuple(trans_primitives_default + agg_primitives): {'ignore_variables': ignore_variables}}

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="df",
                                          max_depth=3, verbose=True, n_jobs=-1,
                                          trans_primitives=trans_primitives,
                                          # primitive_options=primitive_options
                                          )  # MAIN

    feature_matrix: pd.DataFrame = feature_matrix
    print(feature_matrix.head().to_string())
    df = feature_matrix
    df.replace(np.inf, 999999999, inplace=True)
    df.replace(-np.inf, -999999999, inplace=True)
    df.fillna(0, inplace=True)

    df['target'] = df_target
    print(df.head().to_string())
    p = 'feature_eng.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def forest_search_parameters(p):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import cross_val_score
    kfold = StratifiedKFold(n_splits=5)

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop(['target'], 1)
    Y = df['target']
    params = {'n_estimators': [5, 10], 'min_samples_split': [4],
              'max_leaf_nodes': list(range(10, 15)), 'max_depth': list(range(2, 10))}
    clf = GridSearchCV(RandomForestClassifier(), params, cv=kfold)
    results = clf.fit(X, Y)
    print(results.best_estimator_)
    results = cross_val_score(results.best_estimator_, X, Y, cv=kfold)
    print("Accuracy: %f" % results.mean())

# RandomForestClassifier(max_depth=3, max_leaf_nodes=12, min_samples_split=4,
#                        n_estimators=5)

# RandomForestClassifier(max_depth=2, max_leaf_nodes=10, min_samples_split=4,
#                        n_estimators=5)


def feature_importance_forest(df: pd.DataFrame, max_depth=12, n_estimators=25, max_leaf_nodes=14):
    """
    :param df:
    :param max_depth:
    :param n_estimators:
    :param max_leaf_nodes:+5
    """
    # df: pd.DataFrame = pd.read_pickle(p)

    # матрица корреляции
    # переставляем столбцы
    # cols = df.columns.to_list()
    # cols = cols[-1:] + cols[:-1]
    # df = df[cols]

    # import seaborn
    # import matplotlib.pyplot as plt
    #
    # print(df.columns.values)
    # seaborn.heatmap(df.corr(), annot=True, )
    # plt.subplots_adjust(right=1)
    # plt.show()

    X = df.drop(['target'], 1)
    Y = df['target']

    from sklearn.ensemble import RandomForestClassifier

    importance_sum = np.zeros(X.shape[1], dtype=np.float)
    n = 100
    max_depth = np.linspace(2, max_depth+8, 100)  # 12
    n_estimators = np.linspace(5, n_estimators+15, 100)  # 25
    max_leaf_nodes = np.linspace(max_leaf_nodes-4, max_leaf_nodes+8, 100)  # 14
    min_samples_split = 2

    for i in range(n):
        depth = int(round(max_depth[i]))
        n_est = int(round(n_estimators[i]))
        max_l = int(round(max_leaf_nodes[i]))

        model = RandomForestClassifier(random_state=i, max_depth=depth,
                                       n_estimators=n_est, max_leaf_nodes=max_l,
                                       min_samples_split=2)
        model.fit(X, Y)
        # FEATURE IMPORTANCE
        importances = model.feature_importances_  # feature importance
        importance_sum += importances

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importance_sum[indices[f]] / 100))


def corr_matrix(p: str):
    import seaborn
    import matplotlib.pyplot as plt

    df: pd.DataFrame = pd.read_pickle(p)
    print(df.columns.values)
    seaborn.heatmap(df.corr(), annot=True)
    plt.show()


def data_comparision(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # df['Коды отказа'] = df['Коды отказа'].apply(lambda x: x if x >= 0 else 0)
    print(df.shape)
    # print(df.columns.values)
    df1 = df.loc[(df['СФ андерайтера'] == 0) & (df['Коды отказа'] != 97) & (df['Коды отказа'] != 91)]
    df2 = df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97)]  # 4012 in df3
    df3 = df.loc[(df['Статус заявки'] == 1) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97)]
    # print(df3.index)
    df1_i = list(df1.index)
    df2_i = list(df2.index)
    df3_i = list(df3.index)
    res = []
    for i in list(df2.index):
        res.append(i in df3_i)

    print(len(df1_i), len(df2_i), len(df3_i))
    print("sum", sum(res))
    # print(df1)
    # print(df1['Коды отказа'])

    # print(list(df1.index))
    # print(df['Коды отказа'].unique())
    # print(df1)
    import seaborn
    import matplotlib.pyplot as plt

    df: pd.DataFrame = pd.read_pickle(p)
    print(df.columns.values)
    seaborn.heatmap(df.corr(), annot=True)
    plt.show()


def get1(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 1)
    # df1 = df.loc[(df['СФ андерайтера'] == 0) & (df['Коды отказа'] != 97) & (df['Коды отказа'] != 91)]
    df.loc[(df['СФ андерайтера'] == 0) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df


def get2(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 2)
    # df2 = df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97)]  # 4012 in df3
    df.loc[(df['СФ андерайтера'] == 0) | (df['Коды отказа'] == 91) | (df['Коды отказа'] == 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df


def get12(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # 2)
    # df2 = df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97)]  # 4012 in df3
    df.loc[(df['Коды отказа'] == 91) | (df['Коды отказа'] == 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)
    return df


def get3(p: str):
    df: pd.DataFrame = pd.read_pickle(p)
    # print(df.shape)
    # print(df.loc[df['Статус заявки'] == 'Заявка отменена'])
    # # print(df['Статус заявки'])
    # return
    # 3)
    # df3 = df.loc[(df['Статус заявки'] == 1) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97)]
    df.loc[(df['Статус заявки'] == 1) & (df['Коды отказа'] != 91) & (df['Коды отказа'] != 97), "target"] = True
    df['target'].fillna(False, inplace=True)
    # print(df.loc[df['Статус заявки'] == 'Заявка отменена'])
    df.drop('СФ андерайтера', axis=1, inplace=True)
    df.drop('Коды отказа', axis=1, inplace=True)
    df.drop('СФ системы', axis=1, inplace=True)
    df.drop('Статус заявки', axis=1, inplace=True)

    return df


if __name__ == '__main__':
    p = csv_file_read()
    p = impute_v(p)
    p = 'after_imputer.pickle'
    p = encode_categorical(p)

    p = 'encoded.pickle'
    # corr_matrix(p)
    # data_comparision(p)
    # forest_search_parameters()
    # xgboost_serch_parameters()
    # df: pd.DataFrame = pd.read_pickle(p)
    # print(df.iloc[[36141]]['Статус заявки'])
    # print(df.iloc[[2]]['Статус заявки'])
    # print(df.iloc[[3423]]['Статус заявки'])

    # df = get1(p)
    df = get2(p)
    # df = get12(p)
    # df = get3(p)
    # p = feature_engeering(df)
    # p = 'feature_eng.pickle'
    # df: pd.DataFrame = pd.read_pickle(p)
    # forest_search_parameters(p)
    # -- два столбца
    df['Скоринговый балл ОКБ, основной скоринг бюро / Эквифакс 4Score'] = \
        df['Скоринговый балл ОКБ, основной скоринг бюро'] / df['Эквифакс 4Score']
    df.replace([-np.inf], 0, inplace=True)
    df.replace([np.inf], 999999999, inplace=True)
    df['sum'] = df['Оценка кредитной истории Эквифакс'] + df['Скоринговый балл ОКБ, основной скоринг бюро'] + \
                df['Анкетный скоринг'] #+ df['Скоринговый балл НБКИ общ']

    df.dropna(axis=1, inplace=True)
    df.drop('Коды системы', axis=1, inplace=True)
    # feature_importance_forest(df, max_depth=2, max_leaf_nodes=10, n_estimators=5)
    from xgboost_model import xgboost_serch_parameters, feature_importance_xgboost
    # xgboost_serch_parameters(df)
    # print(df)
    feature_importance_xgboost(df)

    # --- анализ
    # df2 = df.loc[(df['Скоринговый балл ОКБ, основной скоринг бюро'] < 1090) & (df['Эквифакс 4Score'] > 548)]
    # df_i = list(df.index)
    # res = []
    # print("good", df2[df2['target'] == True].shape)
    # print("bad", df2[df2['target'] == False].shape)
    # print(df.shape, df2.shape)
    # df2_miss = df.loc[(df['Скоринговый балл ОКБ, основной скоринг бюро'] > 1090) | (df['Эквифакс 4Score'] < 548)]
    # print(df2_miss[df2_miss['target'] == True].shape)
    # print("accuracy", df2[df2['target'] == True].shape[0]/df.shape[0])

    # --- анализ sum<952 and ПВ, %<22 and Скоринговый балл НБКИ общ<569 and Эквифакс 4Score>558
    df2 = df.loc[(df['sum'] < 952) & (df['ПВ, %'] < 22) & (df['Скоринговый балл НБКИ общ'] < 569) & (df['Эквифакс 4Score'] < 558)]
    df_i = list(df.index)
    res = []
    print("positive", df2[df2['target'] == True].shape[0])
    print("false positive", df2[df2['target'] == False].shape[0])
    print(df.shape, df2.shape)
    # df2_miss = df.loc[(df['Скоринговый балл ОКБ, основной скоринг бюро'] > 1090) | (df['Эквифакс 4Score'] < 548)]
    # print("false negative", df2_miss[df2_miss['target'] == True].shape[0])
    print("accuracy", df2[df2['target'] == True].shape[0]/df.shape[0])
    print()
    # --- анализ sum<952 and ПВ, %>60 and Оценка кредитной истории НБКИ>3
    df2 = df.loc[(df['sum'] < 952) & (df['ПВ, %'] > 60) & (df['Оценка кредитной истории НБКИ'] > 3)]
    print("positive", df2[df2['target'] == True].shape[0])
    print("false positive", df2[df2['target'] == False].shape[0])
    print(df.shape, df2.shape)
    # df2_miss = df.loc[(df['Скоринговый балл ОКБ, основной скоринг бюро'] > 1090) | (df['Эквифакс 4Score'] < 548)]
    # print("false negative", df2_miss[df2_miss['target'] == True].shape[0])
    print("accuracy", df2[df2['target'] == True].shape[0] / df.shape[0])