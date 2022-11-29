import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# own
from mart_procs import *
from estimator_xgboost_based import *

from myown_pack.common import impute_v
from myown_pack.common import outliers_numerical
from myown_pack.common import encode_categorical_pipe
from myown_pack.common import standardization
from myown_pack.common import standardization01
from myown_pack.common import feature_selection_cat_rat
from myown_pack.common import condense_category
from myown_pack.common import sparse_classes
from myown_pack.common import fill_na
from myown_pack.model_analysis import drop_collinear_columns
from myown_pack.model_analysis import forest_search_parameters
from myown_pack.model_analysis import xgb_search_parameters
from myown_pack.model_analysis import search_parameters_own_model
from myown_pack.model_analysis import drop_collinear_columns
from myown_pack.exploring import explore_sparse_classes
from myown_pack.exploring import explore_na
from myown_pack.exploring import explore_columns_and_na
from myown_pack.exploring import corr_analysis
from myown_pack.exploring import frequency_analysis
from myown_pack.exploring import shap
from myown_pack.clusterization import *
from myown_pack.plot import *


def autocorrelation_check(p):
    from matplotlib import pyplot as plt
    df: pd.DataFrame = pd.read_pickle(p)
    target = 'ander'
    # X = df.drop([target], 1)
    y = df[target]
    print(y.shape)
    v_random = np.random.choice([0,1], size=(y.shape[0],))
    d = pd.Series(v_random)
    # d = y

    plt.acorr(d.astype(float).tail(60000)[::7], maxlags=50)
    # plt.acorr(d.astype(float), maxlags=20)
    plt.show()
    # pd.plotting.autocorrelation_plot(d.tail(1001)[::7])  # every 10 record
    # plt.show()
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(d.tail(5000), lags=300)
    plt.show()


def outliers_by_hands(p):
    df: pd.DataFrame = pd.read_pickle(p)

    # -- test ids
    ids: list = pd.read_pickle('id_train.pickle')
    print("ids check:")
    assert all(df['id'] == ids)

    # -- select rows
    print('AUTO_DEAL_INITIAL_FEE dropped', df[df['AUTO_DEAL_INITIAL_FEE'] >= 700000].shape[0])
    df = df[df['AUTO_DEAL_INITIAL_FEE'] < 700000]

    print('CLIENT_WI_EXPERIENCE dropped', df[df['CLIENT_WI_EXPERIENCE'] >= 300].shape[0])
    df = df[df['CLIENT_WI_EXPERIENCE'] < 300]  # 58 years
    print('ANKETA_SCORING dropped', df[df['ANKETA_SCORING'] >= 150].shape[0])
    df = df[df['ANKETA_SCORING'] < 150]  # 58 years

    print('CLIENT_DEPENDENTS_COUNT dropped', df[df['CLIENT_DEPENDENTS_COUNT'] >= 3].shape[0])
    df = df[df['CLIENT_DEPENDENTS_COUNT'] < 3]  # 58 years
    print('EQUIFAX_SCORING dropped', df[df['EQUIFAX_SCORING'] <= 100].shape[0])
    df = df[df['EQUIFAX_SCORING'] > 100]  # 58 years

    df = remove_special_cases(df)

    # -- save ids
    save('id_train.pickle', df['id'].tolist())
    # print("ids check:", all(df['id'] == ids))
    # df.reset_index(drop=True, inplace=True)

    return save('outliers_by_hands.pickle', df)


def explore_rej_codes(p):
    df: pd.DataFrame = pd.read_pickle(p)
    print(df.columns.to_list())
    df_cod = df[['DC_REJECTED_CODES', 'DC_REJECTED_CODES_DES']].value_counts()
    print(df_cod.to_string())
    print(df_cod.index['DC_REJECTED_CODES'])
    exit(0)


def explore_second_columns(p):
    df = load(p)
    df = rename_columns(df, fields_dict, len=153)
    # print(df.columns)
    second_cols = [
       'Возраст клиента на дату выдачи кредитного договора',
       'Семейное положение', 'Кол-во иждивенцев', 'Образование',
       'Социальный статус', 'Тип занятости', 'Статус в организации',
       'Стаж в организации (мес.)',
        # 'Доходы, итого',
       # 'Зарплата по основному месту работы',
       #  'Доход клиента',
       #  # 'Пенсия',
       # 'Сдача имущества в аренду',
       #  'Прочие доходы',
       # 'Доходы от частного предпринимательства',
       #  'ПДН на дату выдачи кредита',
                   'Наличие других кредитов на дату выдачи кредита (размер ежемесячный платежей в иные КО)',
                   'Срок кредита контрактный (месяцев)',
                   'Размер кредита (в валюте кредита)', 'Процентная ставка (действ.)',
                   'Тип авто (Б/У, Новый)',
                    'Тип авто (иностранный, отечественный)',
                   'Марка автомобиля','Модель автомобиля', 'Цена АВТО',
                   'Первоначальный взнос / сумму в поле Цена АВТО',
                   # 'Дилер (ТП)',
        # 'Дилер ИНН',
        'Тип дилера (официальный/ не официальный)',
                   'Программа кредитования',
        # 'КАСКО, СК', 'ОСАГО, СК', 'ОСАГО сумма',
        #            'ОСАГО, срок',
        #            'ОСАГО метод оплаты',
        #            'Страхование жизни и здоровья, СК',
                   'Страхование жизни, сумма', 'Страхование жизни и здоровья, срок',
                                               'Страхование жизни и здоровья, тариф',
                   # 'Страхование от потери работы, сумма',
                   # 'Страхование от потери работы, срок',
                   # 'Страхование от потери работы, тариф',
        # 'Ultra24 сумма',
        #            'Ultra24 метод оплаты',
        # 'Оплата за автомобиль, получатель',
                   'Оплата за автомобиль сумма', 'Down Payment Rate (% ПВ)',
                   'Цена автомобиля запрошенная',
        # 'Цена автомобиля (утвержденная)',
                                                  # 'Оплата за доп оборудование сумма',
                   # 'ФИО АНД принявшего последнее решение',
        # 'Модель по ПТС',
        # 'Серия ПТС',
        #            'Телемед +, сумма',
        'PARTNER_ID', 'PARTNER_SHOWROOM_ID',
                   #                                   'Час создания заявки',
                   # 'Месяц создания заявки', 'День недели'
                   ]
    for c in second_cols:
        print(c)
    print(df.columns)
    df2 = df
    for c in second_cols:
        df2 = df2[df2[c].notna()]
        print(c, df2[c].notna().sum())
    # print(df[df['ander'] == 1].tail(10).to_string())
    print(df2.tail(100).to_csv('a.csv'))
    # pd.DataFrame(second_cols).to_csv('a.csv')
    exit()


def pipeline_all():
    """
    auto:
    - boolean to int
    - if object have '12.234' - 3 values we convert it to numeric
    - replace NA for numericla - mean, categorical - most frequent
    - if categorical missing > 0.6 and unique =2 - dont replace NA
    - if  numerical: unique = 2 and has nan - do not fill missing and save as str
    - encode 3-9 - onehot, other - label encoding
    """
    # p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto.csv')  # 10 12 2021 # месяц число год
    # p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto3.csv')  # 12 08 2021
    # p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto4.csv')  # 01 17 2022 jan
    # p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto6.csv')  # 11 feb 2022
    # p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto7.csv')  # 13 feb 2022
    p = csv_file_read('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto8.csv')  # 10 may 2022
    # p = 'after_read.pickle'
    # # exploring(p)
    #
    p = pre_process(p)
    p = 'pre_process.pickle'
    p = process_by_handes(p)
    p = 'by_hands.pickle'
    # # exit()
    # # exploring_rej_codes(p)
    # # exploring_ev(p)
    # # explore_na(p)  # 1 or 0
    # # explore_second_columns(p)

    p1 = 'split_train.pickle'
    p2 = 'split_test.pickle'
    split(p, p1, p2, split_date='2022-02-24')  # and select columns, remove special cases, save id
    #
    p1, p2 = select_and_set_types(p1, p2)
    p1 = 'select_train.pickle'
    p2 = 'select_test.pickle'

    # p1 = outliers_by_hands(p1) # FOR CLUSTERING ONLY
    # p1 = 'outliers_by_hands.pickle'  # evseeva selected
    # frequency_analysis(p1)  # sparse_classes before encoding after sparse_classes HAS EXIT

    p1 = outliers_numerical(p1, 0.0006, target='ander',
                            ignore_columns=['Синтезированный_признак_1', 'Оценка КИ Эквифакс^', 'Оценка КИ ОКБ^',
                                            'Оценка КИ НБКИ^'])  # require fill_na for skew test
    p1 = 'without_outliers.pickle'

    # TODO *IS_CANCEL *CANCEL_DATE - удалить почему нет *_SUM_CONFIRMED - может быть мало и все посмотреть на важность
    # p = feature_selection_cat_rat(p, target='ander', non_na_ration_trash=0.6) # not user with models (we select colmns later)
    # p = 'feature_selection.pickle'

    # explore_sparse_classes(p, min_categories=20, percent=3)
    # explore_columns_and_na(p, fields_dict)
    # replace_names(p)
    # p = 'replace_names.pickle'

    p1 = fill_na(p1, 'fill_na_p1.pickle', p2, 'fill_na_p2.pickle', id_check1='id_train.pickle', id_check2='id_test.pickle')
    # p2 = fill_na(p2, 'fill_na_p2.pickle', id_check='id_test.pickle')
    p1 = 'fill_na_p1.pickle'
    p2 = 'fill_na_p2.pickle'
    p1 = sparse_classes(p1, 'sparse_classes_p1.pickle', id_check='id_train.pickle')
    p2 = sparse_classes(p2, 'sparse_classes_p2.pickle', id_check='id_test.pickle')
    p1 = 'sparse_classes_p1.pickle'
    p2 = 'sparse_classes_p2.pickle'

    # ignore = ['МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр', 'МБКИ_недост', 'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_спецуч',
    #  'МБКИ_паспорт', 'МБКИ_суд', 'МБКИ_тапоМБКИ']
    df1, label_encoders = encode_categorical_pipe(p1, id_check='id_train.pickle', one_hot_min=2,
                                                  onehot_max=10, id_check_f=True)  # 1 or 0 # fill_na required

    df2, label_encoders = encode_categorical_pipe(p2, id_check='id_test.pickle', one_hot_min=2,
                                                  onehot_max=10,
                                                  label_encoders_train=label_encoders, id_check_f=True)  # 1 or 0 # fill_na required
    p1 = 'encoded_p1.pickle'
    p2 = 'encoded_p2.pickle'

    # -- make cols equal
    cols_p1 = df1.columns.to_list()
    cols_p2 = df2.columns.to_list()

    left_cols_p1 = [x for x in cols_p1 if x not in cols_p2]
    if len(left_cols_p1) > 0:
        print("drop p1:", left_cols_p1)
        df1 = df1.drop(columns=left_cols_p1)

    left_cols_p2 = [x for x in cols_p2 if x not in cols_p1]
    if len(left_cols_p2) > 0:
        print("drop p2:", left_cols_p2)
        df2 = df2.drop(columns=left_cols_p2)

    # -- columns in same order
    df2 = df2.reindex(columns=df1.columns.to_list())

    save(p1, df1)
    save(p2, df2)
    assert df2.columns.to_list() == df1.columns.to_list()
    # output: no NA, no strings
    return p1, p2


def pipeline_feature_importance(p: str):
    # -- feature IMPORTANCE --
    # -- forest
    # forest_search_parameters(p, 'ander', n_iter=80, random_state=25)
    # -- selected columns evseeva
    # DecisionTreeClassifier(max_depth=35, max_leaf_nodes=34, min_samples_leaf=9,
    #                        random_state=7)
    # Accuracy: 0.730061
    # AUC: 0.636072
    # Precision: 0.454700
    # Recall: 0.079897
    # Одобренных: 0.044664
    # DecisionTreeClassifier(max_depth=13, max_leaf_nodes=24, min_samples_leaf=7,
    #                        random_state=65)
    # Accuracy: 0.729281
    # AUC: 0.636018
    # Precision: 0.445741
    # Recall: 0.080068
    # Одобренных: 0.045535
    # DecisionTreeClassifier(class_weight={0: 1, 1: 1.25}, max_depth=7,
    #                        max_leaf_nodes=26, min_samples_leaf=4, random_state=65)
    # Accuracy: 0.714862
    # AUC: 0.635146
    # Precision: 0.412152
    # Recall: 0.160365
    # Одобренных: 0.103012
    # RandomForestClassifier(class_weight={0: 1, 1: 0.75}, max_depth=4,
    #                        max_features='sqrt', random_state=25)
    # Accuracy: 0.693205
    # AUC: 0.707775
    # Precision: 0.713333
    # Recall: 0.002527
    # Одобренных: 0.000976

    # -- XGB --------------------------
    # xgb_search_parameters(p, 'ander', n_iter=50, random_state=7, ignore=['id'])
    #
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #               gamma=0, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.01, max_delta_step=0,
    #               max_depth=6, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=200, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=42, reg_alpha=0.5, reg_lambda=0,
    #               scale_pos_weight=1.3377457507636876, seed=42, subsample=0.8,
    #               tree_method='exact', use_label_encoder=False,
    #               validate_parameters=1, verbosity=1)
    # Accuracy: 0.715298
    # AUC: 0.712630
    # Precision: 0.544018
    # Recall: 0.309705
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',
    #               gamma=3, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.1, max_delta_step=0,
    #               max_depth=2, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=430, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=42, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=1.3377457507636876, seed=42, subsample=1,
    #               tree_method='exact', use_label_encoder=False,
    #               validate_parameters=1, verbosity=1)
    # Accuracy: 0.713233
    # AUC: 0.715210
    # Precision: 0.531633
    # # Recall: 0.362968
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',
    #               gamma=3, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.300000012,
    #               max_delta_step=0, max_depth=2, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=260, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=3, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=1.3377457507636876, seed=3, subsample=1,
    #               tree_method='exact', use_label_encoder=False,
    #               validate_parameters=1, verbosity=1)
    # Accuracy: 0.713961
    # AUC: 0.715651
    # Precision: 0.532717
    # Recall: 0.372993
    # -- 3 kfolds
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',
    #               gamma=1, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.300000012,
    #               max_delta_step=0, max_depth=2, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=160, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=22, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=1.3426329555361813, seed=22, subsample=1,
    #               tree_method='exact', use_label_encoder=False,
    #               validate_parameters=1, verbosity=1)
    # Accuracy: 0.711742
    # AUC: 0.717157
    # Precision: 0.525180
    # Recall: 0.384006
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #               gamma=2, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.300000012,
    #               max_delta_step=0, max_depth=3, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=22, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=0.34263295553618134, seed=22, subsample=1,
    #               tree_method='exact', use_label_encoder=False,
    #               validate_parameters=1, verbosity=1)
    # Accuracy: 0.704202
    # AUC: 0.714913
    # Precision: 0.791685
    # Recall: 0.015535
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=0.8, eval_metric='logloss',
    #               gamma=2, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.300000012,
    #               max_delta_step=0, max_depth=2, min_child_weight=6, missing=nan,
    #               monotone_constraints='()', n_estimators=140, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=7, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=1, seed=7, subsample=1, tree_method='exact',
    #               use_label_encoder=False, validate_parameters=1, verbosity=1)
    # Accuracy: 0.729587
    # AUC: 0.672662
    # Precision: 0.504575
    # Recall: 0.157198
    # Одобренных: 0.083276

    # feature_importance_xgboost(p, 'ander')
    pass


def _prepare(p, n_samples=10000):
    from sklearn.utils import resample
    from sklearn.model_selection import train_test_split

    # -- outluers
    df: pd.DataFrame = pd.read_pickle(p)

    # -- ids chech
    ids: list = pd.read_pickle('id.pickle')
    print("ids check:")
    assert all(df['id'] == ids)
    # -- save ids
    # save('id.pickle', df['id'].tolist())

    # p = outliers_numerical(p, 0.002)

    df: pd.DataFrame = pd.read_pickle(p)
    # -- select rows
    df = df.reset_index(drop=True)

    # -- select columns
    cols = [
        'CLIENT_AGE', 'CLIENT_MARITAL_STATUS', 'CLIENT_DEPENDENTS_COUNT',
        'CLIENT_WI_EXPERIENCE', 'ANKETA_SCORING', 'OKB_SCORING',
        'EQUIFAX_SCORING', 'NBKI_SCORING', 'AUTO_DEAL_INITIAL_FEE',
        'MBKI_SCORING',
        # 'МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр',
        # 'МБКИ_недост', 'МБКИ_невыполнена', 'МБКИ_тапоМБКИ',
        'ander',
        'OKB_RATING_SCORING_Нейтральная КИ',
        'OKB_RATING_SCORING_Ошибка выполнения', 'OKB_RATING_SCORING_Плохая КИ',
        'NBKI_RATING_SCORING_Нейтральная КИ',
        'NBKI_RATING_SCORING_Ошибка выполнения',
        'NBKI_RATING_SCORING_Плохая КИ',
        'EQUIFAX_RATING_SCORING_КИ отсутствует',
        'EQUIFAX_RATING_SCORING_Хорошая КИ',
        'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        'EQUIFAX_RATING_SCORING_Плохая КИ'
    ]
    # df = df[cols]
    df = df.drop(columns=['МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр',
                          'МБКИ_недост', 'МБКИ_невыполнена', 'МБКИ_тапоМБКИ', 'МБКИ_розыск',
                          'МБКИ_спецуч', 'МБКИ_суд',
                          'OKB_SCORING',  # not used anymore
                          'OKB_RATING_SCORING_Плохая КИ',  # empy one
                          'OKB_RATING_SCORING_Ошибка выполнения',  # empy one
                          'EQUIFAX_RATING_SCORING_Ошибка выполнения'  # empy one
                          ])

    df_train, df_test = train_test_split(df, test_size=0.8, random_state=2, shuffle=False)
    df = df_test

    df = df.sample(n_samples, random_state=2, replace=False)

    df.reset_index(drop=True, inplace=True)

    # -- standartize
    df = standartize_for_clustering(df, median_mean={'AUTO_DEAL_INITIAL_FEE': 'median',
                                                     'CLIENT_WI_EXPERIENCE': 'mean'},
                                    ignore=['id', 'ander'])
    # -- adjust std for One-Hot encoded columns:
    oh_cols = ['OKB_RATING_SCORING_', 'NBKI_RATING_SCORING_', 'EQUIFAX_RATING_SCORING_']
    df = adjust_for_one_hot_columns(df, oh_cols)

    df.drop(columns=['id']).boxplot()
    plt.xticks(rotation=25)
    plt.show()
    print(df.columns)

    return save("hierarchical_data_prepared.pickle", df)


def _h_look_at_results(p, labels: list):
    target = 'ander'
    df_h: pd.DataFrame = pd.read_pickle(p)
    df_h_o: pd.DataFrame = pd.read_pickle(p)
    df_h = df_h[df_h['ander'] == 1]



    for i in set(labels):
        c_indexes = np.where(np.array(labels) == i)[0].tolist()  # cluster i position

        df_cl: pd.DataFrame = df_h.iloc[c_indexes]  # cluster records
        cl_i = df_cl['id']  # id filed of cluster records
        p = 'sparse_classes.pickle'
        # p = 'encoded.pickle'
        df_o: pd.DataFrame = pd.read_pickle(p)
        df_o.set_index('id', inplace=True)
        # print(cl_i.tolist())
        df_o_cl: pd.DataFrame = df_o.loc[cl_i.tolist()]  # original records for cluster
        print(f'cluster {i} with size {df_cl.shape}')

        # target_values = df_h[target].unique()
        # ot = df_cl[df_cl[target] == target_values[0]].shape[0]
        # od = df_cl[df_cl[target] == target_values[1]].shape[0]
        # print(f"Отклонено/Одобрено - {ot}/{od}")
        # c = 'OKB_RATING_SCORING'
        # print(df[c].value_counts())
        # c = 'MBKI_SCORING'
        # print(df[c].value_counts())
        # print(df_cl['ander'].unique())


        print(df_cl.describe().to_string())
        # print(df_o_cl.describe().to_string())
        # plot_grid_of_plots(df_cl, f'cluster {i}')
        df_o_cl = df_o_cl.drop(columns=['MBKI_SCORING'])
        plot_grid_of_plots(df_o_cl, f'cluster {i}')
        if i == 100:
            # p = 'sparse_classes.pickle'
            p = 'encoded.pickle'
            df_o: pd.DataFrame = pd.read_pickle(p)
            df_o.set_index('id', inplace=True)
            df = df_o.loc[df_h['id'].tolist()]  # original records
            # df = df_o[df_o['id'] in df_h['id']]
            # print(df_h['id'].tolist())
            # print(df_o.index.tolist())
            print(df.describe())

            df['cluster'] = 0  # create column
            df.loc[cl_i, 'cluster'] = 1  # set 1 for cluster
            # print(df_cl.describe().to_string())
            print(df[df['cluster'] == 1].describe().to_string())
            print(df[df['cluster'] == 1]['ander'].sum())
            save('cluster_marks.pickle', df)


            # plot_grid_of_plots(df[df['cluster'] == 1], f'cluster prep {i}')

            # for c in df_o_cl.columns:
            # c = 'MBKI_SCORING'
            # print(c, df_o_cl[c].value_counts().to_string())


def h_clustering(p):

    # p = drop_collinear_columns(p, lim_max=0.92)
    p = 'drop_collinear_columns.pickle'
    p = _prepare(p, n_samples=20000)
    p = "hierarchical_data_prepared.pickle"
    # corr_analysis(p)
    # cluster_loop_pca(p, n_clusters=5)
    # return
    n = 2.7
    # model, labels = hierarchical_clustering(p, distance_threshold=6.1, i=n)
    # hierarchical_clustering_plot(model, 4)
    # save("hier_model.pickle", (model, labels))
    # p = "hier_model.pickle"
    #
    # model, labels = pd.read_pickle(p)
    # p = "hierarchical_data_prepared.pickle"
    # labels = pca_clustering_3(p, i=n, n_clusters=3)
    labels = pca_clustering(p, i=n, n_clusters=4)
    p = "hierarchical_data_prepared.pickle"
    _h_look_at_results(p, labels)


    return
    # h_look_at_results(p, labels)


def h_model(p):
    from xgboost import XGBClassifier

    df = load(p)

    # xgb_search_parameters(p, 'cluster', n_iter=150, random_state=7, ignore=['ander'])

    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #               gamma=2, gpu_id=0, importance_type='gain',
    #               interaction_constraints='', learning_rate=0.300000012,
    #               max_delta_step=0, max_depth=2, min_child_weight=3, missing=np.nan,
    #               monotone_constraints='()', n_estimators=90, n_jobs=2, nthread=2,
    #               num_parallel_tree=1, random_state=7, reg_alpha=0.2, reg_lambda=1,
    #               scale_pos_weight=1, seed=7, subsample=1, tree_method='exact',
    #               use_label_encoder=False, validate_parameters=1, verbosity=1)
    m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
                      gamma=2, gpu_id=0, importance_type='gain',
                      interaction_constraints='', learning_rate=0.300000012,
                      max_delta_step=0, max_depth=2, min_child_weight=6, missing=np.nan,
                      monotone_constraints='()', n_estimators=130, n_jobs=2, nthread=2,
                      num_parallel_tree=1, random_state=7, reg_alpha=0.5, reg_lambda=1,
                      scale_pos_weight=1, seed=7, subsample=1, tree_method='exact',
                      use_label_encoder=False, validate_parameters=1, verbosity=1
                      )
    # -- train
    target = 'cluster'
    X: pd.DataFrame = df.drop([target], 1)
    X = df.drop(['ander', target], 1)
    y = df[target]
    X = StandardScaler().fit_transform(X)  # XGB specific
    check_model_sklearn_cross(m, X, y)

    m.fit(X, y)
    check_model_sklearn_split(m, X, y)
    print(" -- test")
    p = 'encoded.pickle'
    df = load(p)
    X: pd.DataFrame = df.drop(['ander', 'id'], 1)
    y = df['ander']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
                                                        random_state=1, stratify=None)

    train_rate = sum(y_train) / (len(y_train) - sum(y_train))
    X_test, y_test = downsample(X_test, y_test, rate=train_rate)
    m.predict(X_test)
    check_model_sklearn_split(m, X_test, y_test)


def model_explore(p1, p2):
    """ on hold out """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    target = 'ander'
    # -- data
    df = load(p2)
    print(df.columns)
    X_test_o: pd.DataFrame = df.drop(columns=[target])
    y_test_o = df[target]

    df_train = load(p1)
    y_train = df_train[target]

    train_rate = sum(y_train) / (len(y_train) - sum(y_train))
    test_rate = sum(y_test_o) / (len(y_test_o) - sum(y_test_o))
    rate = (train_rate + test_rate) / 2
    print("train_rate", train_rate, "test_rate", test_rate, "rate", rate)
    X_test, y_test = downsample(X_test_o, y_test_o, rate=rate)
    # print(X_test.describe().to_string())
    # exit()
    print("X_test.shape", X_test.shape)
    print("y_test.shape", sum(y_test))

    print("Всего:", df.shape)
    print("Отклонено", df[df[target] == 0].shape)
    print("Одобрено", df[df[target] == 1].shape)

    # return


    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #                   gamma=2, gpu_id=0, importance_type='gain',
    #                   interaction_constraints='', learning_rate=0.300000012,
    #                   max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
    #                   monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
    #                   num_parallel_tree=1, random_state=22, reg_alpha=0.2, reg_lambda=1,
    #                   scale_pos_weight=0.984263295553618134, seed=22, subsample=1,
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
    #                   importance_type='gain', interaction_constraints='',
    #                   learning_rate=0.02, max_delta_step=0, max_depth=6,
    #                   min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #                   n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
    #                   random_state=42, reg_alpha=0.5, reg_lambda=1,
    #                   scale_pos_weight=0.75, seed=42, subsample=1, # 1.3346062351320898
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.8, eval_metric='logloss',
    #                   gamma=2, gpu_id=0, importance_type='gain',
    #                   interaction_constraints='', learning_rate=0.300000012,
    #                   max_delta_step=0, max_depth=2, min_child_weight=6, missing=np.nan,
    #                   monotone_constraints='()', n_estimators=140, n_jobs=2, nthread=2,
    #                   num_parallel_tree=1, random_state=7, reg_alpha=0.2, reg_lambda=1,
    #                   scale_pos_weight=1, seed=7, subsample=1, tree_method='exact',  # 0.7 - with otcl
    #                   use_label_encoder=False, validate_parameters=1, verbosity=1)
    # m = RandomForestClassifier(class_weight={0: 1, 1: 1}, max_depth=6,
    #                            max_features='sqrt', n_estimators=80, random_state=7)
    # m = DecisionTreeClassifier(class_weight={0: 1, 1: 0.588},  # criterion='entropy',
    #                            max_depth=60, max_leaf_nodes=56, min_samples_leaf=7,
    #                            min_weight_fraction_leaf=0, random_state=5)
    # m = DecisionTreeClassifier(max_depth=35, max_leaf_nodes=34, min_samples_leaf=9,
    #                        random_state=7, class_weight={0: 1, 1: 1}) #)
    #
    # m = MyEstimXGB(v1a=-0.325, v1b=0.2, v1c=0.0, v2a=0.65, v2b=240.0, v2c=2.3,
    #                v3a=63.75, v3b=-0.05, v3c=0.2, v4a=-0.25, v4b=0.1, v5a=0.25,
    #                v5b=-0.2, v5c=0.36, v6a=0, v6b=0, v7a=0.01, v7b=-0.1)
    m = MyEstimXGB(v1a=-0.32, v1b=0.175, v1c=0.02333333333333333,
                   v2a=0.0028666666666666667, v2b=2.466666666666667, v3a=61.0,
                   v3b=-0.02, v3c=0.23, v4a=-0.26, v4b=0.03266666666666667,
                   v5a=0.21333333333333332, v5b=-0.2, v5c=0.36,
                   v6a=0.0010333333333333334, v6b=0.84, v6c=-0.06666666666666667,
                   v7a=0.9, v71b=-0.18, v72b=-0.07, v73b=0.1)
    # m.set_params()  # required for MyEstimXGB

    y_pred = m.predict(X_test)
    print("y_pred", len(y_pred), sum(y_pred))

    i_pred = np.where(y_pred == 1)[0]
    print("i_pred", len(i_pred))
    i_rej = np.where(y_pred == 0)[0]
    print("i_rej", len(i_rej))
    appr = X_test.iloc[i_pred]
    rej = X_test.iloc[i_rej]
    p = 'by_hands.pickle'
    df_o = load(p)
    # print(len(y_pred), sum(y_pred))
    # a["id"]
    df_o.set_index('id', inplace=True)
    # print(cl_i.tolist())
    # print(df_o.columns)
    # print(appr["id"].to_list())
    # print(df_o)
    # print(df_o.loc[appr["id"].to_list()]['ander'].value_counts())
    # return
    # print(appr["id"].sort_values())
    df_appr: pd.DataFrame = df_o.loc[appr["id"].to_list()]
    df_rej: pd.DataFrame = df_o.loc[rej["id"].to_list()]
    df_rej_fail: pd.DataFrame = df_rej[df_rej['ander'] == 1]

    df_all: pd.DataFrame = df_o.loc[X_test_o["id"].to_list()]

    def rej_cods():
        print("df_all", df_all.shape)
        df_all.loc[appr["id"].to_list(), ['model_appr']] = 1
        df_all['model_appr'] = df_all['model_appr'].fillna(0)

        print("df_all", df_all.shape)
        print("model_appr", df_all['model_appr'].sum())
        print("ander", df_all['ander'].sum())
        print(len(appr["id"].to_list()))
        print(df_all['model_appr'].describe())
        print(df_all['model_appr'].sum())
        df_all.to_csv('test.csv')
        df_appr = df_all[df_all['ander'] == 1]  # [(df_all['ander'] == 1) & (df_all['model_appr'] == 0)]
        df_rej = df_all[df_all['ander'] == 0]  # [(df_all['ander'] == 0) & (df_all['model_appr'] == 1)]

        df_appr_c = df_appr['DC_REJECTED_CODES1'].value_counts().head(10)
        df_rej_c = df_rej['DC_REJECTED_CODES1'].value_counts().head(10)
        df_cod = df_appr[['DC_REJECTED_CODES', 'DC_REJECTED_CODES_DES']].value_counts().head(20)
        print(df_cod.head(20))
        df_cod2 = df_rej[['DC_REJECTED_CODES', 'DC_REJECTED_CODES_DES']].value_counts().head(20)
        print(df_cod2.head(20))
        x = np.arange(len(df_appr_c.index))
        width = 0.35
        plt.bar(x - width/2, df_appr_c.index)
        indx = sorted(list(set(df_rej_c.index.tolist()).union(set(df_appr_c.index.tolist()))))
        print(indx)

        a = []
        r = []
        for i in indx:
            if i in df_appr_c:
                a.append(df_appr_c.loc[i])
            else:
                a.append(0)
            if i in df_rej_c:
                r.append(df_rej_c.loc[i])
            else:
                r.append(0)
        pd.DataFrame({"Ошибочно отклоненные": a,
                      "Ошибочно одобренные": r}, index=indx).plot.bar(rot=15, figsize=(10, 5))

        plt.title("Тестовой выборке ошибочно отклоненные и ошибочно одобренные")
        # plt.show()
        plt.savefig("Тестовой выборке ошибочно отклоненные и ошибочно одобренные")

    # rej_cods()


    # print(df_appr['ander'].shape, sum(df_appr['ander']))

    print("all appoved by_model ander=1", df_appr.shape)
    # df_appr = df_appr[df_appr['DEAL_STATUS'] == 'cancel']
    # print(df_appr['DEAL_CREATED_DATE'])
    # df_cod = df_appr[['DC_REJECTED_CODES', 'DC_REJECTED_CODES_DES']].value_counts()
    # print(df_cod.to_string())
    # print(df_appr.index.sort_values())
    # print(df_appr['DEAL_CREATED_DATE'].sort_values())
    # print(df_appr['DC_REJECTED_CODES'].unique())
    # print()

    from download import download_mart_oracle

    l = df_appr[df_appr['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    l_and = df_all[df_all['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    l_rej = df_rej[df_rej['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    l_rej_fail = df_rej_fail[df_rej_fail['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    print("released approved by model", len(l))
    print("released approved by ander", len(l_and))
    print("released rej by model, appr by ander", len(l_rej))
    print(f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l[50:]))}')
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                         sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l[:50]))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred1: pd.DataFrame = pd.read_csv(p)
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l[50:]))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred2: pd.DataFrame = pd.read_csv(p)
    df_cred = pd.concat([df_cred1, df_cred2], ignore_index=True)
    print("df_cred.shape", df_cred.shape)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()
    print("approved:")
    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)

    # --original prosr
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l_and[:50]))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred1: pd.DataFrame = pd.read_csv(p)
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l_and[50:]))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred2: pd.DataFrame = pd.read_csv(p)
    df_cred = pd.concat([df_cred1, df_cred2], ignore_index=True)
    print("df_cred.shape", df_cred.shape)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()
    print("and:")
    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)

    # -- rej fail
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l_rej_fail))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred: pd.DataFrame = pd.read_csv(p)
    print("df_cred.shape", df_cred.shape)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()
    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)

    x = [1, 2, 3]
    y_and = [6257, 828, 32]
    y_m = [1515, 132, 5]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y_and)  #
    ax.fill_between(x, 0, y_and, facecolor='red')
    ax.plot(x, y_m)  #
    ax.fill_between(x, 0, y_m, facecolor='green')
    plt.show()

    exit()

    # print(df_cred.to_string())

    # print(df_cred[df_cred['CNT_PROSR_ALL'] != 0].describe().to_string())
    df_appr = df_appr['DEAL_STATUS'].value_counts()
    pros_has_l = [0 for _ in range(df_appr.shape[0])]
    pros_long_has_l = pros_has_l.copy()

    pros_has_l[df_appr.index.tolist().index('release')] = pros_has

    pros_long_has_l[df_appr.index.tolist().index('release')] = pros_long_has


    # df_appr['DEAL_STATUS'].value_counts().plot.bar(rot=15)
    print(df_appr.to_list(), df_appr.index)

    pd.DataFrame({"Статус заявки": df_appr.loc['release'],
                  "Просрочка больше 1 дня": pros_has,
                  "Непрерывная Просрочка больше 5 дней": pros_long_has}, index=['release']).plot.bar(rot=15)

    plt.title(f"Одобренные моделью на тесте {len(l)}. Выпущенные.png")
    plt.savefig(f"Одобренные моделью на тесте {len(l)}. Выпущенные.png")
    plt.close()


    # -- в тестовых данных без модели



    df_o = df_o.sort_values(by=['DEAL_CREATED_DATE'])
    X_train, X_test = train_test_split(df_o, test_size=0.20, shuffle=False,
                                       random_state=1, stratify=None)
    print(X_train.shape, X_test.shape)

    a = X_test[X_test['ander'] == 1]['DEAL_STATUS'].value_counts()

    # a['DEAL_STATUS'].value_counts().plot.bar(rot=15)

    X_test = X_test[X_test['ander'] == 1]
    l = X_test[X_test['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    print(len(l))
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred: pd.DataFrame = pd.read_csv(p)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()

    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)
    pd.DataFrame({"Статус заявки": a.loc['release'],
                  "Просрочка больше 1 дня": pros_has,
                  "Непрерывная Просрочка больше 1 дня": pros_long_has}, index=['release']).plot.bar(rot=15)
    plt.title(f"Всего на тесте {len(l)} Выпущенные")
    plt.savefig(f"Всего на тесте.png")
    plt.close()

    X_train = X_train[X_train['ander'] == 1]
    l = X_train[X_train['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist()
    print(len(l))
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in {str(tuple(l))}',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'])
    df_cred: pd.DataFrame = pd.read_csv(p)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()

    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)
    pd.DataFrame({"Статус заявки": a.loc['release'],
                  "Просрочка больше 1 дня": pros_has,
                  "Непрерывная Просрочка больше 1 дня": pros_long_has}, index=['release']).plot.bar(rot=15)
    plt.title(f"Всего на обучающей {len(l)} Выпущенные")
    plt.savefig(f"Всего на обучающей.png")
    plt.close()
    # plt.show()

    # print("all X_test ander=1:", a.shape)
    # print(a['DEAL_CREATED_DATE'].sort_values())
    # # print(a['DEAL_CREATED_DATE'])
    # # print(a[228229])
    # # a = a[a['DEAL_STATUS'] == 'cancel']
    # df_cod = a[['DC_REJECTED_CODES', 'DC_REJECTED_CODES_DES']].value_counts()
    # # print(df_cod.to_string())
    # print(a['DC_REJECTED_CODES'].unique())
    # print(a['DC_REJECTED_CODES1'].unique())
    # # print(df_cod.index['DC_REJECTED_CODES'])
    # # pd.DataFrame({'s1': s1, 's2': s2}).plot.bar(rot=15)
    # plt.title(f"all x_test. Значения DEAL_STATUS:")
    # plt.show()

    # print(X_test.shape, df_appr.shape)


def prosroch_explore(m, p2, target):
    df_test: pd.DataFrame = pd.read_pickle(p2)
    X_test: pd.DataFrame = df_test.drop([target, 'id'], 1)
    y_test = df_test[target]

    y_pred = m.predict(X_test)
    print("y_pred", len(y_pred), "sum(y_pred)", sum(y_pred))
    i_pred = np.where(y_pred == 1)[0]
    print("i_pred", len(i_pred))
    print("y_pred", len(y_pred), sum(y_pred))
    appr = df_test.iloc[i_pred]
    p = 'by_hands.pickle'
    df_o = load(p)
    df_o.set_index('id', inplace=True)
    df_appr: pd.DataFrame = df_o.loc[appr["id"].to_list()]
    from download import download_mart_oracle
    l = set(df_appr[df_appr['DEAL_STATUS'] == 'release']['DEAL_ID'].tolist())
    print("released appro  ved by model", len(l))
    p = download_mart_oracle("mart_cred_sel.pickle", table_name='MART_CRED',
                             sql=f'SELECT DEAL_ID,CNT_PROSR_ALL,CNT_LONG_PROSR FROM ldw.MART_CRED WHERE DEAL_ID in ',
                             cols=['DEAL_ID', 'CNT_PROSR_ALL', 'CNT_LONG_PROSR'],
                             l=list(l))
    df_cred: pd.DataFrame = pd.read_csv(p)
    print("df_cred.shape", df_cred.shape)
    df_cred['pros_has'] = df_cred['CNT_PROSR_ALL'].apply(lambda x: x > 0)
    df_cred['pros_long_has'] = df_cred['CNT_LONG_PROSR'].apply(lambda x: x > 5)
    df_cred = df_cred.groupby(by='DEAL_ID').sum()
    pros_has = df_cred['pros_has'].apply(lambda x: int(x > 0)).sum()
    pros_long_has = df_cred['pros_long_has'].apply(lambda x: int(x > 0)).sum()
    print("approved:")
    print("Просрочка больше 1 дня", pros_has)
    print("Непрерывная Просрочка больше 5 дней", pros_long_has)


def best2(p1,p2):
    """ find best classification approach - by comparing different classificators"""
    df_train = load(p1)
    df_test = load(p2)
    # -- fix types
    df_train = df_train.astype(float)
    df_test = df_test.astype(float)

    df_train.drop(columns=["id"], inplace=True)
    df_test.drop(columns=["id"], inplace=True)

    d = [
        # 'CLIENT_AGE',
        # 'CLIENT_GENDER',
        # 'CLIENT_MARITAL_STATUS', 'CLIENT_DEPENDENTS_COUNT', 'CLIENT_WI_EXPERIENCE',
        # 'ANKETA_SCORING', 'OKB_SCORING', 'EQUIFAX_SCORING', 'NBKI_SCORING', 'AUTO_DEAL_INITIAL_FEE', 'МБКИ_треб_адрес',
        # 'МБКИ_треб_исп_пр', 'МБКИ_треб_пассп', 'МБКИ_требаналотч', 'МБКИ_нет_огр', 'МБКИ_недост', 'МБКИ_невыполнена',
        # 'МБКИ_данные_не', 'День недели',
        #
        # 'OKB_RATING_SCORING_КИ отсутствует', 'OKB_RATING_SCORING_Нейтральная КИ',
        # 'OKB_RATING_SCORING_Ошибка выполнения', 'OKB_RATING_SCORING_Плохая КИ', 'OKB_RATING_SCORING_Хорошая КИ',
        # 'NBKI_RATING_SCORING_КИ отсутствует', 'NBKI_RATING_SCORING_Нейтральная КИ',
        # 'NBKI_RATING_SCORING_Ошибка выполнения', 'NBKI_RATING_SCORING_Хорошая КИ',
        # 'EQUIFAX_RATING_SCORING_КИ отсутствует', 'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        # 'EQUIFAX_RATING_SCORING_Ошибка выполнения', 'EQUIFAX_RATING_SCORING_Хорошая КИ',

        # '0prov_status_code', '0prov_status_type', '0prov_status_groupid',
        '0prov_status_isActing',
        # '0prov_companyType', '0prov_okfs_code', '0prov_includeInList', '0prov_legalAddresses',
        # 'ex4scor_scoring', 'afsnbki_scoring', 'okbnh_MA_SMT', 'okbnh_MA_EMP', 'okbnh_MA_REF', 'okbnh_MA_MTE',
        # 'okbnh_MA_SPA', 'okbnh_MA_SPE', 'okbnh_MULT_M', 'okbnh_MA_MS', 'okbnh_MA_SAM', 'okbnh_MA_AS_', 'okbnh_LCL_MA',
        # 'okbnh_MA_MS_', 'okbnh_MA_RAD', 'okbnh_MA_PAS', 'okbnh_MA_SWT', 'okbnh_MA_AS', 'nbki_biom_resp_matchImages',
        # 'nbki_biom_resp_matchResults', 'nbki_biom_resp_match_max', 'nbki_biom_resp_match_avg',
        # '0prov_status_groupName_Действующее', '0prov_status_groupName_Реорганизация'
    ]

    # df_train.drop(columns=d, inplace=True)
    # df_test.drop(columns=d, inplace=True)
    print(df_train.isna().sum().to_string())
    # exit()

    from sklearn.model_selection import StratifiedShuffleSplit, KFold
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    import xgboost as xgb
    from sklearn.feature_selection import SelectFromModel

    classifiers = [
        KNeighborsClassifier(3),
        SVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression()
    ]

    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    df_train = df_train.tail(2000)

    target = 'ander'
    X_train: pd.DataFrame = df_train.drop(columns=[target])
    X = X_train.to_numpy()
    X_test: pd.DataFrame = df_test.drop(columns=[target])

    y_train = df_train[target]
    y = y_train.to_numpy()
    y_test = df_test[target]

    lr = LinearDiscriminantAnalysis()
    lr.fit(X_train, y_train)
    yprd = lr.predict(X_test)
    acc1 = accuracy_score(y_test, yprd)
    print(acc1 * 100)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc
                # print ('{0}: {1}'.format(name, acc * 100))

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    # plt.show()
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    from operator import itemgetter

    sorted_dict = sorted(acc_dict.items(), key=itemgetter(1), reverse=True)

    for k, v in sorted_dict:
        print("{0}-{1:.2%}".format(k, v))

    ntrain = X_train.shape[0]
    ntest = y_test.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS)

    class SklearnHelper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def fit(self, x, y):
            return self.clf.fit(x, y)

        def feature_importances(self, x, y):
            print(self.clf.fit(x, y).feature_importances_)

    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    # Put in our parameters for said classifiers
    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }

    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    lr_params = {}

    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)

    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test)  # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, X_train, y_train, X_test)  # AdaBoost
    gb_oof_train, gb_oof_test = get_oof(gb, X_train, y_train, X_test)  # Gradient Boost
    lr_oof_train, lr_oof_test = get_oof(lr, X_train, y_train, X_test)  # Logistic Regression

    print("Training is complete")

    base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                           'ExtraTrees': et_oof_train.ravel(),
                                           'AdaBoost': ada_oof_train.ravel(),
                                           'GradientBoost': gb_oof_train.ravel(),
                                           'LogisticRegression': lr_oof_train.ravel()
                                           })

    # data = [
    #     go.Heatmap(
    #         z=base_predictions_train.astype(float).corr().values,
    #         x=base_predictions_train.columns.values,
    #         y=base_predictions_train.columns.values,
    #         colorscale='Portland',
    #         showscale=True,
    #         reversescale=True
    #     )
    # ]
    # py.iplot(data, filename='labelled-heatmap')

    x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, lr_oof_train), axis=1)
    x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, lr_oof_test), axis=1)

    gbm = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=36,
        max_depth=5,
        min_child_weight=1,
        # gamma=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(x_train, y_train)
    y_pred = gbm.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Fit model using each importance as a threshold

    # thresholds = sort(model.feature_importances_)
    thresholds = sorted(gbm.feature_importances_, reverse=True)

    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(gbm, threshold=thresh, prefit=True)
        select_X_train = selection.transform(x_train)
        # train model
        selection_model = xgb.XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

    # Achieves accuracy of 77.78% using Xgboost


def drop_cols(p1, p2):
    dr = [
        # 'EQUIFAX_SCORING',
        'ex4scor_scoring',  # corr with EQUIFAX_SCORING
        # 'CLIENT_WI_EXPERIENCE',
        # 'OKB_RATING_SCORING_КИ отсутствует',
        # 'afsnbki_scoring',
        # 'AUTO_DEAL_INITIAL_FEE',
        # 'OKB_RATING_SCORING_Хорошая КИ',
        # 'NBKI_SCORING',
        # 'CLIENT_MARITAL_STATUS',
        # 'ANKETA_SCORING',
        # 'nbki_acc_payAsAgreed_without_cc',
        # 'МБКИ_требаналотч',
        # 'МБКИ_нет_огр',
        # 'okb_count_acc_ipoteka',
        # 'OKB_SCORING',
        # 'equifax_acc_payAsAgreed_without_cc',
        # 'okbnh_MA_SPA',
        # 'all_acc_payAsAgreed',
        'all_acc_payAsAgreed_without_cc', # corr with all_acc_payAsAgreed
        'equifax_acc_payAsAgreed',  # corr with equifax_acc_payAsAgreed_without_cc
        'all_count_acc_ipoteka',  # corr with okb_count_acc_ipoteka
        # 'okb_count_acc_higher50_closed',
        'all_count_acc_higher50_closed',  # corr with okb_count_acc_higher50_closed
        # 'CLIENT_GENDER',
        # 'nbki_count_acc_without_credit_cards',
        # 'okb_acc_payAsAgreed',
        'mbki_isMassAddress_yes',  # corr with МБКИ_треб_адрес
        # 'NBKI_RATING_SCORING_КИ отсутствует',
        # 'nbki_acc_payAsAgreed',
        # 'nbki_biom_resp_matchImages',
        # 'nbki_count_acc_credit_cards',
        # 'okb_count_acc_credit_cards',
        'okb_acc_payAsAgreed_without_cc',  # corr with okb_acc_payAsAgreed
        # 'okbnh_MA_SAM',
        # 'NBKI_RATING_SCORING_Хорошая КИ',
        # 'okb_count_acc_without_credit_cards',
        # 'okbnh_MA_RAD',
        # 'equifax_acc_late29Days_without_cc',
        # 'nbki_count_acc_higher50_closed',
        # 'nbki_biom_resp_match_max',
        'okb_count_total',  # corr with okb_count_acc_credit_cards
        # 'CLIENT_AGE',
        'nbki_biom_resp_matchResults',  # corr with afsnbki_scoring
        # 'День недели',
        # 'all_count_acc_without_credit_cards',
        # 'МБКИ_треб_исп_пр',
        # 'МБКИ_данные_не',
        # 'all_count_acc_lower50_closed',
        'all_count_total',  # corr with all_count_acc_without_credit_cards
        'mbki_inBanksStopLists',  # corr with МБКИ_требаналотч
        # 'nbki_acc_late29Days_without_cc',
        'nbki_acc_late29Days', #    corr with nbki_acc_late29Days_without_cc
        # 'okbnh_LCL_MA',
        'nbki_count_total',  # corr with nbki_count_acc_credit_cards
        # 'EQUIFAX_RATING_SCORING_КИ отсутствует',
        # 'okbnh_MULT_M',
        # 'equifax_count_acc_without_credit_cards',
        # 'all_acc_late29Days',
        'EQUIFAX_RATING_SCORING_Хорошая КИ',  # corr with NBKI_RATING_SCORING_Хорошая КИ
        'equifax_acc_late29Days',  # corr with equifax_acc_late29Days_without_cc
        # 'CLIENT_DEPENDENTS_COUNT',
        'okbnh_MA_SPE',  # corr with okbnh_MA_SAM
        # 'mbki_isInnConfirmed',
        'all_acc_late29Days_without_cc',  # corr with 'all_acc_late29Days'
        # 'МБКИ_треб_адрес',
        # 'INITIAL_FEE_DIV_CAR_PRICE',
        'equifax_count_acc_ipoteka',
        'equifax_count_total',
        'mbki_isMassAddress_another',
        'nbki_count_acc_lower50_closed',
        # 72
        'nbki_count_acc_ipoteka',
        'okb_count_acc_lower50_closed',
        'OKB_RATING_SCORING_Нейтральная КИ',
        'okb_acc_late29Days',
        'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        'МБКИ_недост',
        'all_count_acc_credit_cards',
        'NBKI_RATING_SCORING_Нейтральная КИ',
        'okb_acc_late29Days_without_cc',
        'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        'okbnh_MA_SMT',
        # 'mbki_hasAdministrativeCrimesData_yes', # very rare - special case
        'okbnh_MA_SWT',
        'okbnh_MA_MTE',
        'OKB_RATING_SCORING_Ошибка выполнения',
        'mbki_isMassLeader',
        'mbki_informationNotFound',
        'okbnh_MA_REF',
        'okbnh_MA_AS',
        'okbnh_MA_MS',
        'OKB_RATING_SCORING_Плохая КИ',
        'МБКИ_треб_пассп',
        'okbnh_MA_PAS',
        # 'mbki_hasAdministrativeCrimesData_another', # very rare - special case
        'okbnh_MA_AS',
        'okbnh_MA_EMP',
        'okbnh_MA_MS',
        'МБКИ_невыполнена',
        'NBKI_RATING_SCORING_Ошибка выполнения',
        'equifax_count_acc_higher50_closed',
        'equifax_count_acc_lower50_closed',
        'equifax_count_acc_credit_cards',
    ]

    old_own = [
        'CLIENT_AGE',
        'CLIENT_MARITAL_STATUS',
        'CLIENT_DEPENDENTS_COUNT',
        'CLIENT_WI_EXPERIENCE',
        'ANKETA_SCORING',
        'OKB_SCORING',
        'EQUIFAX_SCORING',
        'NBKI_SCORING',
        'AUTO_DEAL_INITIAL_FEE',
        'МБКИ_треб_адрес',
        'МБКИ_треб_исп_пр',
        'МБКИ_треб_пассп',
        'МБКИ_требаналотч',
        'МБКИ_нет_огр',
        'МБКИ_недост',
        'mbki_isMassAddress_another',  # 'МБКИ_розыск',
        'МБКИ_невыполнена',
        'mbki_isMassAddress_yes',  # 'МБКИ_налспецуч',
        'МБКИ_данные_не',
        'День недели',
        'OKB_RATING_SCORING_КИ отсутствует',
        'OKB_RATING_SCORING_Хорошая КИ',
        'OKB_RATING_SCORING_Нейтральная КИ',
        'OKB_RATING_SCORING_Ошибка выполнения',
        'OKB_RATING_SCORING_Плохая КИ',
        'NBKI_RATING_SCORING_Хорошая КИ',
        'NBKI_RATING_SCORING_Нейтральная КИ',
        'NBKI_RATING_SCORING_КИ отсутствует',
        'NBKI_RATING_SCORING_Ошибка выполнения',
        'EQUIFAX_RATING_SCORING_КИ отсутствует',
        'EQUIFAX_RATING_SCORING_Хорошая КИ',
        'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        'ander', "id",
        'INITIAL_FEE_DIV_CAR_PRICE',
        'AUTO_DEAL_COST_REQUESTED'

    ]

    df1 = load(p1)
    df2 = load(p2)

    print(df1.columns.to_list())
    df1 = df1[df1['id'] > 100000]
    print(df1.shape)

    df1.drop(columns=dr, inplace=True)
    df2.drop(columns=dr, inplace=True)
    # df1 = df1[old_own]
    # df2 = df2[old_own]
    print(df1.columns.to_list())

    p1 = save('encoded_p1_2.pickle', df1)
    p2 = save('encoded_p2_2.pickle', df2)
    return p1, p2


if __name__ == '__main__':
    pd.options.mode.use_inf_as_na = True
    # p1, p2 = pipeline_all()
    p1 = 'encoded_p1.pickle'
    p2 = 'encoded_p2.pickle'
    p1, p2 = drop_cols(p1, p2)
    p1 = 'encoded_p1_2.pickle'
    p2 = 'encoded_p2_2.pickle'


    # p = rename_columns(p, fields_dict, 99)
    # p = 'renamed.pickle'
    # best2(p1,p2)
    # box_ander(p1)
    # box_ander_pand(p)
    # p = 'box_ander.pickle'
    # box_ander_interpret(p)
    # histogram_two_in_one(p=p1, feature_main='AUTO_DEAL_INITIAL_FEE', feature_binary='ander')
    # histogram_two_in_one(p=p1, feature_main='INITIAL_FEE_DIV_CAR_PRICE', feature_binary='ander')
    #--  p = impute_v(p)
    #--  p = 'after_imuter.pickle'


    # p = 'drop_collinear_columns.pickle'
    # p = drop_collinear_columns(p1, lim_max=0.98, drop=[], method='spearman')  # and 1 or 0 ander
    # p = 'drop_collinear_columns.pickle'
    # -- feature importance
    # pipeline_feature_importance(p)

    # -- analisys
    # corr_analysis(p1, method="pearson")
    # corr_analysis(p2, method="kendall")
    # linear_analysis(p)  # after encoded and drop_collinear_columns
    # plot_x_linear(p) # fail
    # model_search(p)
    # xgb_search_parameters(p1, p2, target='ander', n_iter=60)
    # forest_search_parameters(p1, 'ander', n_iter=60, random_state=25)
    # autocorrelation_check(p)

    # print(a)
    # print(a.map(fields_dict, na_action='ignore'))
    # [print(x) for x in df.columns]

    # df:pd.DataFrame= pd.read_pickle(p1)
    # for c in df.columns:
    #     if df[c].isna().sum() != 0
    #         print(c, df[c].isna().sum())
    # # exit()
    # # print(df.)
    #
    # exit()
    # p = feature_importance_composite(p1, 'ander')  # outliers_numerical feature_selection fill_na sparse_classes encode_categorical
    p = feature_importance_composite_permut(p1, 'TT', nrep_forst=3, nrep_xgboost=2)
    p = 'feature_importance_comp.pickle'
    feature_importance_composite_plot(p)
    # im2, count = feature_importance_xgboost_static(p1, 'ander', perm=True, nrep=6)

    # pair_scatter_plot(p1)
    # m = test_model_own(p1, p2)
    # m = search_parameters_own_model(MyEstimXGB_wz(), p1, p2, 'ander', n_iter=3000, random_state=5)
    # prosroch_explore(m, p2, target='ander')

    # test_model_own_dtree_cases(p)
    # test_model(p1, p2)  # , random_state=46
    # test_model_regression(p)
    # test_model_decision_tree(p)
    # test_model_change_graph(p1, p2)
    # model_explore(p1, p2)
    # shap(p1)
    # frequency_analysis(p)
    # h_clustering(p)
    # p = 'cluster_marks.pickle'
    # h_model(p)


    # os.system('curl --data chat_id="-461260910" --data "text=Completed" "https://api.telegram.org/bot1990575229:AAHWtk8fGJoKXtYCyE_K43JWlAtkQg9dGp0/sendMessage"')
