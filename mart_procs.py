import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# own

from myown_pack.plot import rename_columns
from mart_fields import fields_dict
from myown_pack.common import save, load
from myown_pack.common import downsample
from myown_pack.common import remove_single_unique_values
from myown_pack.model_analysis import permutation_importance_forest
from myown_pack.plot import make_confusion_matrix
from myown_pack.model_analysis import check_model_sklearn_split, check_model_sklearn_cross
from mart_procs_scorings import add_scoring_resp


def csv_file_read(p):
    # -- get main table
    df = pd.read_csv(p, index_col=0, low_memory=False)

    # # -- get status target
    # df_and: pd.DataFrame = pd.read_pickle('deal_id_ander.pickle')
    # df_and.set_index('id заявки', verify_integrity=True, inplace=True)
    # df_and = df_and.sort_index()
    # print(df_and)
    # df = df.join(df_and)
    # print(df.shape)

    df['DEAL_CREATED_DATE'] = pd.to_datetime(df['DEAL_CREATED_DATE'])
    df = df.sort_values(by=['DEAL_CREATED_DATE']).reset_index(drop=True)
    # -- for merge with scorings
    BEGIN = '2020-01-01'
    END = '2022-05-10'
    print(f"deleted rows 'DEAL_CREATED_DATE'] < {BEGIN}):", df[df['DEAL_CREATED_DATE'] < BEGIN].shape[0])
    print(f"deleted rows 'DEAL_CREATED_DATE'] > {END}):", df[df['DEAL_CREATED_DATE'] > END].shape[0])

    df = df[(df['DEAL_CREATED_DATE'] > '2020-01-01')]
    # df = df[(df['DEAL_CREATED_DATE'] < '2022-02-10')]  # year , month, date
    # df = merge_nbki_rating(df)

    return save('after_read.pickle', df)


def remove_special_cases(df):
    deleted = df[df['МБКИ_недост'] == 1].shape[0]
    if 'МБКИ_треб_пассп' in df.columns.tolist():
        deleted += df[df['МБКИ_треб_пассп'] == 1].shape[0]
    deleted += df[df['МБКИ_данные_не'] == 1].shape[0]
    if 'МБКИ_розыск' in df.columns.tolist():
        deleted += df[df['МБКИ_розыск'] == 1].shape[0]
    if 'МБКИ_налспецуч' in df.columns.tolist():
        deleted += df[df['МБКИ_налспецуч'] == 1].shape[0]
    print("deleted special_cases: ", deleted)

    df = df[df['МБКИ_недост'] != 1]
    if 'МБКИ_треб_пассп' in df.columns.tolist():
        df = df[df['МБКИ_треб_пассп'] != 1]
    df = df[df['МБКИ_данные_не'] != 1]
    if 'МБКИ_розыск' in df.columns.tolist():
        df = df[df['МБКИ_розыск'] != 1]
    if 'МБКИ_налспецуч' in df.columns.tolist():
        df = df[df['МБКИ_налспецуч'] != 1]
    return df


def exploring(p):
    df: pd.DataFrame = pd.read_pickle(p)
    from myown_pack.exploring import corr_analysis
    corr_analysis(p, method="pearson")
    exit()

    # df['ander'] = df[fds]
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Одобрено'), 'ander'] = 'Одобрено'
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Отмена'), 'ander'] = 'Отмена'

    print(df['ander'])
    # df[fds]
    a = df.value_counts(subset=['CLIENT_AGE', 'ander'], dropna=False)
    print(a)
    print()
    a = df.value_counts(subset=['ander', 'CLIENT_AGE'], dropna=False)
    print(a)
    exit()
    # [df['CLIENT_AGE']

    # print(df)
    # df = df[df['DEAL_CREATED_DATE'] < '2021-04-28']
    # df = df[df['DEAL_CREATED_DATE'] > '2019-11-20']
    # print('Одобрено', df[df[fds] == 'Одобрено'].shape)
    # print('Отмена', df[df[fds] == 'Отмена'].shape)
    # print('код отмены 01 091',
    #       df[(df['DC_REJECTED_CODES'] == '01') | df['DC_REJECTED_CODES'].str.startswith('091', na=False)].shape)
    #
    # df = df[(df[fds] == 'Одобрено') | (df[fds] == 'Отмена')]
    # # print(df['DEAL_REACHED_UNDER'])
    # # print(df['DEAL_REACHED_UNDER'].unique())
    # # print(df[df['DEAL_REACHED_UNDER'] == 0].shape)
    # # print(df[df['DEAL_REACHED_UNDER'] == 1].shape)
    # print(df[df['PRC_UNDER_FIO']].isna().sum().shape)

    # print(df['DEAL_STATUS'].unique())
    # df = df[df['DEAL_STATUS'] == 'release']

    # print(df[df['DEAL_ID']!= pd.NA ].shape)
    # df.dropna(axis='index', subset=['DEAL_ID'], inplace=True)
    # df.set_index(deal_id_field, verify_integrity=True, inplace=True)
    # df = df.sort_index()
    # print(df['DEAL_ID'].isna().sum())
    # print("shape", df.shape)
    # print(df.columns.to_list())
    # # print(df['DECISION_DEAL'].unique())
    # # print(df[df['DEAL_ID'] != pd.NA].tail(100).to_string())
    # print(df[df['DECISION_DEAL'] == 'Одобрено'])
    # print(df.columns.values)
    # print(df['DC_REJECTED_CODES'].unique())
    # print(df['SF_UNDER'].unique())
    print(df.dtypes.to_string())


def exploring_rej_codes(p):
    df: pd.DataFrame = pd.read_pickle(p)

    df = df[df['DEAL_CREATED_DATE'] > '2021-09-03']

    plt.plot_date(df['DEAL_CREATED_DATE'], df['DC_REJECTED_CODES_COUNT'])
    plt.xticks(rotation=10)
    plt.show()
    exit()


def exploring_ev(p):
    df: pd.DataFrame = pd.read_pickle(p)

    c = 'CLIENT_AGE'
    df = df[(df[c] > 37) & (df[c] < 50)]
    print(c, df.shape[0])
    c = 'CLIENT_MARITAL_STATUS'
    df = df[df[c] == 'в браке']
    print(c, df.shape[0])
    c = 'CLIENT_DEPENDENTS_COUNT'
    df = df[df[c] > 0]
    print(c, df.shape[0])
    c = 'CLIENT_WI_EXPERIENCE'
    df = df[df[c] > 3 * 12]
    print(c, df.shape[0])
    c = 'ANKETA_SCORING'
    df = df[df[c] > 65]
    print(c, df.shape[0])

    c = 'OKB_SCORING'
    import matplotlib.pyplot as plt
    # df[c].plot.hist()
    # plt.savefig("")
    df = df[(df[c] == 0) | (df[c] > 870)]
    print(c, df.shape[0])
    c = 'EQUIFAX_SCORING'
    df = df[(df[c] == 0) | (df[c] > 760)]
    print(c, df.shape[0])
    c = 'NBKI_SCORING'
    df = df[(df[c] == 0) | (df[c] > 770)]
    print(c, df.shape[0])
    c = 'AUTO_DEAL_INITIAL_FEE'
    df = df[df[c] > 20]
    print(c, df.shape[0])
    c1 = 'NBKI_RATING_SCORING'
    c2 = 'OKB_RATING_SCORING'
    c3 = 'EQUIFAX_RATING_SCORING'
    df = df[((df[c1] == 'Хорошая КИ') | (df[c2] == 'Хорошая КИ') | (df[c3] == 'Хорошая КИ'))]
    # одна или две отсутствуют
    # одна или две нейтральная
    # если одна плохая - не берем

    print(c1, c2, c3, df.shape[0])

    # print(df[df['ander'] == 0].shape[0])  # Одобрено
    # print(df[df['ander'] == 1].shape[0])
    # print(df[df['ander'] == 2].shape[0])

    print("Одобрено", df[df['ander'] == 0].shape[0])  # Одобрено
    print("Отклонено", df[df['ander'] == 1].shape[0])
    print("Отклонено пользователем", df[df['ander'] == 2].shape[0])

    df[df['ander'] == 0].to_csv("Одобрено.csv", na_rep=' ')
    df[df['ander'] == 1].to_csv("Отклонено.csv", na_rep=' ')
    df[df['ander'] == 2].to_csv("Отклонено пользователем 01 091.csv", na_rep=' ')


def _box_ander_calc(df, box: dict, cols: tuple):
    # box = {'CLIENT_AGE':(2,3), 'CLIENT_MARITAL_STATUS':(3,4)}
    for ci in box.keys():
        c = cols[ci]
        gr1, gr2 = box[ci]
        df = df[(gr1 <= df[c]) & (df[c] < gr2)]
    return df[df['ander'] == 1].shape[0], df[df['ander'] == 0].shape[0]


def _a(i, box_up: dict, cols, boxes: dict, df: pd.DataFrame) -> list:
    """ """
    if i > len(cols) - 1:
        re = _box_ander_calc(df, box_up, cols)
        return [(box_up, re)]
    c = cols[i]
    res = []
    for j, b in enumerate(boxes[c]):
        if i == 0 or i == 1:
            print(box_up)
        box = box_up.copy()
        box[i] = b
        b_a = _a(i + 1, box, cols, boxes, df)
        res.extend(b_a)
    return res


# --- numpy
def _box_ander_calc_n(df, box: np.ndarray, cols: tuple):
    # box = {'CLIENT_AGE':(2,3), 'CLIENT_MARITAL_STATUS':(3,4)}
    for i, r in enumerate(box):
        gr1, gr2 = r
        c = cols[i]
        df = df[(gr1 <= df[c]) & (df[c] < gr2)]
    return df[df['ander'] == 1].shape[0], df[df['ander'] == 0].shape[0]


import time


# -- numpy
def _a_n(i, box_up: np.ndarray, cols, boxes: dict, df: pd.DataFrame) -> list:
    """ box_up - rows = len(cols)
     boxes = {col:[(gr1,gr2),...]"""
    if i > len(cols) - 1:
        re = _box_ander_calc_n(df, box_up, cols)
        # print(re)
        return [(box_up, re)]
    c = cols[i]
    res = []
    for j, b in enumerate(boxes[c]):
        if j == 0 and i < 5:
            print(i, j)
        # time.sleep(0.001)
        # if i == 0 and j == 0:
        # print(i,j, box_up)
        box = box_up.copy()
        box[i] = (b[0], b[1])
        b_a = _a_n(i + 1, box, cols, boxes, df)
        res.extend(b_a)
    return res


# -- pandas
def _box_ander_calc_pand(df: np.ndarray, box: np.ndarray):
    # box = {'CLIENT_AGE':3, 'CLIENT_MARITAL_STATUS':2}
    # ['range2', 'range4', 'range13', 'range42', 'range79', 'range83']
    # b = '(range2 == {}) & (range4 == {})  & (range13 == {}) & (range42 == {}) & (range79 == {}) & (range83 == {})'.format(*box.tolist())  # & (range12 == {})  & (range84 == {}) & (range57 == {}
    # df = df.query(b)
    # df = df[(df.iloc[:, 0] == box[0]) & (df.iloc[:, 1] == box[1]) & (df.iloc[:, 2] == box[2]) & (df.iloc[:, 3] == box[3])
    #       & (df.iloc[:, 4] == box[4]) & (df.iloc[:, 5] == box[5]) & (df.iloc[:, 6] == box[6])]
    df = df[(df[:, 0] == box[0]) &
            (df[:, 1] == box[1]) &
            (df[:, 2] == box[2]) &
            (df[:, 3] == box[3]) &
            (df[:, 4] == box[4]) &
            (df[:, 5] == box[5]) &
            (df[:, 6] == box[6])
            & (df[:, 7] == box[7])
        # & (df[:, 8] == box[8])
        # & (df[:, 9] == box[9])
        # & (df[:, 10] == box[10])
        # (df[:, 11] == box[11]) &
        # (df[:, 12] == box[12]) &
        # (df[:, 13] == box[13])
        # (df[:, 14] == box[14])
            ]

    return df[df[:, -1] == 1].shape[0], df[df[:, -1] == 0].shape[0]  # 'ander


def _a_pand(i, box_up: np.ndarray, cols: int, boxes: list, df: pd.DataFrame) -> list:
    """ box_up - rows = len(cols)
     boxes = {col:[(gr1,gr2),...]"""

    if i > cols - 1:
        # calc count in box
        box = box_up
        df = df[(df[:, 0] == box[0]) &
                (df[:, 1] == box[1]) &
                (df[:, 2] == box[2]) &
                (df[:, 3] == box[3]) &
                (df[:, 4] == box[4]) &
                (df[:, 5] == box[5]) &
                (df[:, 6] == box[6])
                & (df[:, 7] == box[7])
                & (df[:, 8] == box[8])
                & (df[:, 9] == box[9])
                ]
        appr = df[df[:, -1] == 1].shape[0]
        rej = df[df[:, -1] == 0].shape[0]
        # appr, rej = _box_ander_calc_pand(df, box_up)
        return [(box_up, appr, rej)]

    res = []
    for j, b in enumerate(boxes[i][1]):
        if i < 3:  # and i < 5:
            print(i, j, box_up)
        box = box_up.copy()
        box[i] = j
        b_a = _a_pand(i + 1, box, cols, boxes, df)
        # free memory
        b_a2 = []
        for x in b_a:
            box_up, appr, rej = x
            if appr > 1 or rej > 1:
                b_a2.append(x)

        res.extend(b_a2)
    return res


def box_ander(p: str):
    df: pd.DataFrame = pd.read_pickle(p)

    cols = (
        # 'CLIENT_AGE',
        'CLIENT_MARITAL_STATUS',
        #     'CLIENT_DEPENDENTS_COUNT',
        'CLIENT_WI_EXPERIENCE',
        #     'ANKETA_SCORING',
        'OKB_SCORING',
        'EQUIFAX_SCORING',
        'NBKI_SCORING',
        'AUTO_DEAL_INITIAL_FEE',
        #     'OKB_RATING_SCORING_Хорошая КИ',
        'OKB_RATING_SCORING_КИ отсутствует',
        #     'OKB_RATING_SCORING_Нейтральная КИ',
        #     'OKB_RATING_SCORING_Ошибка выполнения',
        #     'OKB_RATING_SCORING_Плохая КИ',
        'NBKI_RATING_SCORING_Хорошая КИ',
        'NBKI_RATING_SCORING_КИ отсутствует',
        #     'NBKI_RATING_SCORING_Нейтральная КИ',
        #     'NBKI_RATING_SCORING_Плохая КИ',
        #     'NBKI_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Хорошая КИ',
        #     'EQUIFAX_RATING_SCORING_КИ отсутствует',
        #     'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Плохая КИ',
        #     'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        #     'MBKI_SCORING'
        # 'МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр', 'МБКИ_недост', 'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_спецуч',
        # 'МБКИ_суд',
        'МБКИ_тапоМБКИ'
    )
    # print(df.columns.to_list())
    # return
    #
    # return

    # from itertools import combinations
    # perm = combinations(cols, len(cols))
    # print(list(perm))
    # exit(0)
    bins_count = 3
    # c1_bin1.2.2.3.
    # c2_bin1.2.3.4
    # res = { columns : box_arr
    boxes = dict()
    for c in cols:
        # print(c)
        if len(df[c].unique()) == 2:
            box_arr = ((0, 1), (1, 2))
            # print(_box_ander_calc(df, {c: (0, 1)}))
            # print(_box_ander_calc(df, {c: (1, 2)}))
        else:
            box_arr_r = np.histogram_bin_edges(df[c], bins=bins_count)  # 'rice'
            print(c, len(box_arr_r))
            box_arr = []
            for i in range(len(box_arr_r) - 1):
                box_arr.append((box_arr_r[i], box_arr_r[i + 1]))
                # print(_box_ander_calc(df, {c: (box_arr[i], box_arr[i + 1])}))

        boxes[c] = box_arr
    # print(boxes)
    # return
    boxes_full = []
    # box = {'CLIENT_AGE':(2,3), 'CLIENT_MARITAL_STATUS':(3,4)}
    # indexes = [0 for i in range(len(cols))]
    # if c1 == c2:
    #     continue
    # box = {}
    # c1 = cols[0]
    # for b in boxes[c1]:
    #     box1 = box.copy()
    #     box1[c1] = b
    #
    #     c2 = cols[1]
    #     for b in boxes[c2]:
    #         box2 = box1.copy()
    #         box2[c2] = b
    #
    #         c3 = cols[2]
    #         for b in boxes[c3]:
    #             box3 = box2.copy()
    #             box3[c3] = b
    # import time
    import concurrent.futures

    # r = _a(0, {}, cols, boxes, df)
    empty_box = np.zeros((len(cols), 2))
    # r = _a_n(0, empty_box, cols, boxes, df)
    # [print(x) for x in r]
    # return
    # --- multithread
    i = 0
    c = cols[i]
    res = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tsks = []
        for j, b in enumerate(boxes[c]):
            # if i == 0 or i == 1:
            #     print(box_up)
            box = empty_box.copy()
            box[i] = (b[0], b[1])
            t = executor.submit(_a_n, i + 1, box, cols, boxes, df)
            # t = _a(i + 1, box, cols, boxes, df)
            tsks.append(t)
        for t in tsks:
            b_a = t.result()

            res.extend(b_a)

    print("fin")
    # [print(x) for x in res]

    # [(rowcomb,(appr, rej))...]
    return save('box_ander', res)


def no_rec_func(indexes, indexes_end, df, lens):
    lenght = len(lens)
    box = indexes
    df_sel = df[(df[:, 0] == box[0]) &
                (df[:, 1] == box[1]) &
                (df[:, 2] == box[2]) &
                (df[:, 3] == box[3]) &
                (df[:, 4] == box[4]) &
                (df[:, 5] == box[5]) &
                (df[:, 6] == box[6])
                & (df[:, 7] == box[7])
                & (df[:, 8] == box[8])
                & (df[:, 9] == box[9])
                & (df[:, 10] == box[10])
                & (df[:, 11] == box[11])
        # & (df[:, 12] == box[12])
                ]
    res = []
    while indexes != indexes_end:
        changes = False
        for i, x in enumerate(reversed(indexes)):  # i = 0,1,2,3, x = 9,8,7,6,5
            if x >= lens[lenght - i - 1]:
                indexes[lenght - i - 1] = 0
                indexes[lenght - i - 2] += 1
                if i > lenght - 5:
                    print(indexes)
                if i > lenght - 8:  # 10-7-1 = - change in first 7
                    changes = True

        if changes:  # calc all
            # calc box, columns dependent:
            box = indexes
            df_sel = df[(df[:, 0] == box[0]) &
                        (df[:, 1] == box[1])
                        & (df[:, 2] == box[2])
                        & (df[:, 3] == box[3])
                        & (df[:, 4] == box[4])
                        & (df[:, 5] == box[5])
                        & (df[:, 6] == box[6])
                        ]
            df_sel_fin = df_sel[(df_sel[:, 7] == box[7])
                                & (df_sel[:, 8] == box[8])
                                & (df_sel[:, 9] == box[9])
                                & (df_sel[:, 10] == box[10])
                                & (df_sel[:, 11] == box[11])
                # & (df_sel[:, 12] == box[12])
                                ]
        else:  # calc for last one
            indexes[lenght - 1] += 1
            box = indexes
            df_sel_fin = df_sel[(df_sel[:, 7] == box[7])
                                & (df_sel[:, 8] == box[8])
                                & (df_sel[:, 9] == box[9])
                                & (df_sel[:, 10] == box[10])
                                & (df_sel[:, 11] == box[11])
                # & (df_sel[:, 12] == box[12])
                                ]

        appr = df_sel_fin[df_sel_fin[:, -1] == 1].shape[0]
        rej = df_sel_fin[df_sel_fin[:, -1] == 0].shape[0]

        if appr > 2 or rej > 2:
            res.append((indexes.copy(), appr, rej))

    return res


def box_ander_pand(p: str):
    import concurrent.futures
    df: pd.DataFrame = pd.read_pickle(p)
    # df = df.iloc[:1000]
    df = df[df['CLIENT_WI_EXPERIENCE'] < 200]
    cols = (
        #     'CLIENT_DEPENDENTS_COUNT',
        # -- long
        'CLIENT_WI_EXPERIENCE',  # 4
        'ANKETA_SCORING',  # 9
        'OKB_SCORING',  # 7

        # 'MBKI_SCORING',  # 13
        # 'CLIENT_AGE',  # 14
        'CLIENT_MARITAL_STATUS',  # 11
        'OKB_RATING_SCORING_Хорошая КИ',  # 2
        'OKB_RATING_SCORING_КИ отсутствует',  # 1
        #     'OKB_RATING_SCORING_Нейтральная КИ',
        #     'OKB_RATING_SCORING_Ошибка выполнения',
        #     'OKB_RATING_SCORING_Плохая КИ',
        'NBKI_RATING_SCORING_Хорошая КИ',  # 10
        'NBKI_RATING_SCORING_КИ отсутствует',  # 8
        #     'NBKI_RATING_SCORING_Нейтральная КИ',
        #     'NBKI_RATING_SCORING_Плохая КИ',
        #     'NBKI_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Хорошая КИ',
        #     'EQUIFAX_RATING_SCORING_КИ отсутствует',
        #     'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Плохая КИ',
        #     'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        # 'МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр', 'МБКИ_недост', 'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_спецуч',
        # 'МБКИ_суд',
        'МБКИ_тапоМБКИ',  # 12
        # -- long
        'EQUIFAX_SCORING',  # 3
        'NBKI_SCORING',  # 5
        'AUTO_DEAL_INITIAL_FEE',  # 6
    )

    bins_count = 10
    boxes = []  # {c:[(0,1),(1,2)..], ...}
    for c in cols:
        if len(df[c].unique()) == 2:
            box_arr = ((0, 1), (1, 2))
        else:
            box_arr_r = np.histogram_bin_edges(df[c], bins=bins_count)  # 'rice'
            print(c, len(box_arr_r))
            box_arr = []
            for i in range(len(box_arr_r) - 1):
                box_arr.append((box_arr_r[i], box_arr_r[i + 1]))
        # ci = df.columns.get_loc(c)
        boxes.append((c, box_arr))
    print(boxes)
    # return
    # create columns with range mark
    for cr in boxes:
        c, ranges = cr
        ci = df.columns.get_loc(c)
        for i, r in enumerate(ranges):
            gr1, gr2 = r
            df.loc[(gr1 <= df[c]) & (df[c] < gr2), 'range' + str(ci)] = i

    # r = _a(0, {}, cols, boxes, df)
    cols2 = [c for c in df.columns if c.startswith('range')]
    print(len(cols2), cols2)

    df = df[cols2 + ['ander']]

    # -- no recursion
    # one time:
    lenght = len(cols)
    indexes = [0] * lenght
    indexes_end = [0] * lenght
    for i in range(lenght):
        indexes_end[i] = len(boxes[i][1]) - 1
    # first box:
    # appr, rej = _box_ander_calc_pand(df.to_numpy(), indexes)  # empty first

    lens = [len(x[1]) for x in boxes]  # lens per column
    df = df.to_numpy()

    res_full = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tsks = []
        for ii in range(lens[0]):
            indexes2 = indexes.copy()
            indexes_end2 = indexes_end.copy()
            indexes2[0] = ii
            indexes_end2[0] = ii
            print(indexes2)
            print(indexes_end2)
            t = executor.submit(no_rec_func, indexes2, indexes_end2, df, lens)
            tsks.append(t)
        for t in tsks:
            r = t.result()
            print("wtf11", r)
            res_full.extend(r)
    # res_full = no_rec_func(indexes, indexes_end, df, lens)
    return save('box_ander.pickle', res_full)

    # r = _a_pand(0, empty_box, len(cols) , boxes, df)
    # [print(x) for x in r]
    # return
    # --- multithread
    i = 0
    # c = cols[i]
    cols = len(cols)
    res = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tsks = []
        for j, b in enumerate(boxes[i][1]):
            box = empty_box.copy()
            box[i] = j
            t = executor.submit(_a_pand, i + 1, box, cols, boxes, df.to_numpy())
            tsks.append(t)
        for t in tsks:
            b_a = t.result()
            res.extend(b_a)

    print("fin", len(res))

    return save('box_ander.pickle', res)


def box_ander_interpret(p: str):
    cols = (
        #     'CLIENT_DEPENDENTS_COUNT',
        # -- long
        'CLIENT_WI_EXPERIENCE',  # 4
        'ANKETA_SCORING',  # 9
        'OKB_SCORING',  # 7

        # 'MBKI_SCORING',  # 13
        # 'CLIENT_AGE',  # 14
        'CLIENT_MARITAL_STATUS',  # 11
        'OKB_RATING_SCORING_Хорошая КИ',  # 2
        'OKB_RATING_SCORING_КИ отсутствует',  # 1
        #     'OKB_RATING_SCORING_Нейтральная КИ',
        #     'OKB_RATING_SCORING_Ошибка выполнения',
        #     'OKB_RATING_SCORING_Плохая КИ',
        'NBKI_RATING_SCORING_Хорошая КИ',  # 10
        'NBKI_RATING_SCORING_КИ отсутствует',  # 8
        #     'NBKI_RATING_SCORING_Нейтральная КИ',
        #     'NBKI_RATING_SCORING_Плохая КИ',
        #     'NBKI_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Хорошая КИ',
        #     'EQUIFAX_RATING_SCORING_КИ отсутствует',
        #     'EQUIFAX_RATING_SCORING_Ошибка выполнения',
        #     'EQUIFAX_RATING_SCORING_Плохая КИ',
        #     'EQUIFAX_RATING_SCORING_Нейтральная КИ',
        # 'МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр', 'МБКИ_недост', 'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_спецуч',
        # 'МБКИ_суд',
        'МБКИ_тапоМБКИ',  # 12
        # -- long
        'EQUIFAX_SCORING',  # 3
        'NBKI_SCORING',  # 5
        'AUTO_DEAL_INITIAL_FEE',  # 6
    )
    binary = ['OKB_RATING_SCORING_Хорошая КИ', 'OKB_RATING_SCORING_КИ отсутствует',
              'NBKI_RATING_SCORING_КИ отсутствует']
    not_binary = ['EQUIFAX_SCORING', 'NBKI_SCORING', 'AUTO_DEAL_INITIAL_FEE']
    boxes = [('CLIENT_WI_EXPERIENCE', [(0.0, 19.9),
                                       (19.9, 39.8),
                                       (39.8, 59.699999999999996),
                                       (59.699999999999996, 79.6),
                                       (79.6, 99.5),
                                       (99.5, 119.39999999999999),
                                       (119.39999999999999, 139.29999999999998),
                                       (139.29999999999998, 159.2), (159.2, 179.1), (179.1, 199.0)]), ('ANKETA_SCORING',
                                                                                                       [(35.0, 49.8),
                                                                                                        (49.8, 64.6),
                                                                                                        (64.6, 79.4),
                                                                                                        (79.4, 94.2),
                                                                                                        (94.2, 109.0), (
                                                                                                            109.0,
                                                                                                            123.80000000000001),
                                                                                                        (
                                                                                                            123.80000000000001,
                                                                                                            138.60000000000002),
                                                                                                        (
                                                                                                            138.60000000000002,
                                                                                                            153.4),
                                                                                                        (153.4,
                                                                                                         168.20000000000002),
                                                                                                        (
                                                                                                            168.20000000000002,
                                                                                                            183.0)]), (
                 'OKB_SCORING',
                 [(-111.0, 20.0), (20.0, 151.0), (151.0, 282.0), (282.0, 413.0), (413.0, 544.0), (544.0, 675.0),
                  (675.0, 806.0), (806.0, 937.0), (937.0, 1068.0), (1068.0, 1199.0)]),
             ('CLIENT_MARITAL_STATUS', ((0, 1), (1, 2))), ('OKB_RATING_SCORING_Хорошая КИ', ((0, 1), (1, 2))),
             ('OKB_RATING_SCORING_КИ отсутствует', ((0, 1), (1, 2))),
             ('NBKI_RATING_SCORING_Хорошая КИ', ((0, 1), (1, 2))),
             ('NBKI_RATING_SCORING_КИ отсутствует', ((0, 1), (1, 2))), ('МБКИ_тапоМБКИ', ((0, 1), (1, 2))), (
                 'EQUIFAX_SCORING',
                 [(0.0, 95.2), (95.2, 190.4), (190.4, 285.6), (285.6, 380.8), (380.8, 476.0), (476.0, 571.2),
                  (571.2, 666.4), (666.4, 761.6), (761.6, 856.8000000000001), (856.8000000000001, 952.0)]),
             ('NBKI_SCORING',
              [(0.0, 85.0),
               (85.0, 170.0),
               (
                   170.0, 255.0),
               (
                   255.0, 340.0),
               (
                   340.0, 425.0),
               (
                   425.0, 510.0),
               (
                   510.0, 595.0),
               (
                   595.0, 680.0),
               (
                   680.0, 765.0),
               (765.0,
                850.0)]), (
                 'AUTO_DEAL_INITIAL_FEE',
                 [(0.0, 190000.0), (190000.0, 380000.0), (380000.0, 570000.0), (570000.0, 760000.0),
                  (760000.0, 950000.0),
                  (950000.0, 1140000.0), (1140000.0, 1330000.0), (1330000.0, 1520000.0), (1520000.0, 1710000.0),
                  (1710000.0, 1900000.0)])]
    res: list = pd.read_pickle(p)
    # [print(x) for x in res]
    # return
    res = [(
        x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8], x[0][9], x[0][10], x[0][11],
        x[1], x[2]) for x in res]
    df = pd.DataFrame(res, columns=list(cols) + ['appr', 'rej'])
    df['diff'] = df['appr'] - df['rej']  # + round((200 - df['rej'])/3)
    # df['diff'] = df['appr'] - df['rej']  + round((200 - df['rej'])/3)
    df.sort_values(by=['diff'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    print(df.head(30).to_string())
    # print(df[df['diff'] >= 3].to_string())
    return
    # return
    # print(df.head(30).sort_values(by=binary).to_string())
    # return
    # print(df.head(30).sort_values(by=['EQUIFAX_SCORING', 'NBKI_SCORING', 'AUTO_DEAL_INITIAL_FEE']).to_string())
    # return
    # --one row
    # ind = 0
    # r = df.iloc[[ind]]
    # for i, c in enumerate(cols):
    #     bs = boxes[i][1]
    #     print(c, bs[int(r[c][0])])
    # --range rows

    for ii, r in enumerate(df[df['diff'] >= 4].iterrows()):
        # s:pd.Series = r[1]
        # print(s.loc['OKB_SCORING'])
        # print(s)
        # return
        print(f'#---- {ii + 1} ------')  # str(r[1].loc['diff'])
        print('elif')
        for i, c in enumerate(cols):
            boxes_per_column = boxes[i][1]  # boxes for column
            # print(r, c)
            range = boxes_per_column[int(r[1].loc[c])]
            # print(c, range)
            if c in binary:
                if range == (0, 1):
                    print("and  r['" + c + "']  == 0" + ' \\')
                else:
                    print("and  r['" + c + "']  == 1" + ' \\')
            else:
                print('and ' + str(range[0]) + " < r['" + c + "'] < " + str(range[1]) + ' \\')
        print(':')
        print('res = 1')

    # r = df.iloc[[ind]]
    # for i, c in enumerate(cols):
    #     bs = boxes[i][1]
    #     print(c, bs[int(r[c][0])])


def pre_process(p):
    """ select rows we require, prepare major columns, join dataframes"""
    df: pd.DataFrame = pd.read_pickle(p)

    # df[df['CLIENT_DEPENDENTS_COUNT'] != 0]['CLIENT_DEPENDENTS_COUNT'].hist(bins=30)
    # print(df.columns.tolist())
    # print(df)
    # exit()

    # df['c'] = df[df['CLIENT_AGE'].isna()]
    # --------------------- SHAPOVALOV
    # print(df.shape)
    # df_shapovalov: pd.DataFrame = pd.read_csv('/home/u2/Desktop/df_ml_v3.csv')
    # df_shapovalov.drop(columns=['FIRST_DECISION_STATE'], inplace=True)
    # df = pd.merge(df,df_shapovalov, on='APP_SRC_REF', how="inner", indicator=False)
    # -------------------------------
    # print(df['2.949'].unique())
    # fs = df[(df['2.949'] == 1) | (df['2.949'] == 2)]
    # # print(
    # print(fs.shape)
    #
    # print(fs[fs['FIRST_DECISION_STATE'] == 'Одобрено'])
    # exit()
    # df = df[(df['DEAL_CREATED_DATE'] > '2021-03-12')]
    # print(df.head(100).to_string())
    # print(df[df['FIRST_DECISION_STATE'] == 'Одобрено'].shape)
    # print(df[df['FIRST_DECISION_STATE'] == 'Отмена'].shape)
    # print()
    # exit()

    print(df.columns.tolist())
    # print(df.head(61).to_string())
    print(df.shape)
    # exit()

    # -- SET ID
    df['id'] = df['APP_SRC_REF']

    # fds = 'FIRST_DECISION_STATE'
    fds = 'FIRST_DECISION_STATE_ALL'

    # -- observe
    print("all", df.shape)
    print('Одобрено', df[(df['DEAL_REACHED_UNDER'] == 1) & (df[fds] == 'Одобрено')].shape)
    print('Отмена', df[(df['DEAL_REACHED_UNDER'] == 1) & (df[fds] == 'Отмена')].shape)
    # print(df[(df['DEAL_REACHED_UNDER'] == 1) & (df[fds] == 'Отмена'))
    # print(df[(df['DEAL_REACHED_UNDER'] == 1) & (df[fds] == 'Одобрено')])
    # return
    # -- удаляем лишние строки под дате
    # print("last date:", df['DEAL_CREATED_DATE'].sort_values().tail(1))
    # 2019-10-31

    # --
    print()
    print("first date:", df['DEAL_CREATED_DATE'].sort_values().head(1))
    print("last date:", df['DEAL_CREATED_DATE'].sort_values().tail(1))
    print()
    # X_train, X_test = train_test_split(df, test_size=0.20, shuffle=False,
    #                                    random_state=1, stratify=None)
    # print("BEST train date:", X_train['DEAL_CREATED_DATE'].sort_values().tail(1))
    # print("test shape", X_test.shape)
    # print('test Одобрено', X_test[X_test[fds] == 'Одобрено'].shape)
    # print('test Отмена', X_test[X_test[fds] == 'Отмена'].shape)
    # print()

    #  --    target
    # Если код отказа 06 и 'DEAL_REACHED_UNDER' == 1, то решение андеррайтера можно считать полноценным (Коринев м)
    df['ander'] = df[fds]
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Одобрено'), 'ander'] = 'Одобрено'
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Отмена'), 'ander'] = 'Отмена'

    c01 = ((df['DC_REJECTED_CODES'] == '01') | df['DC_REJECTED_CODES'].str.startswith('01;', na=False) | df[
        'DC_REJECTED_CODES'].str.endswith('01', na=False))
    c091 = ((df['DC_REJECTED_CODES'] == '091') | df['DC_REJECTED_CODES'].str.startswith('091;', na=False) | df[
        'DC_REJECTED_CODES'].str.endswith('091', na=False))
    # Отменено пользователем, одобрено андером
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Одобрено') &
           (c01 | c091), 'ander'] = 'Отменено пользователем, одобрено андером'
    # Отменено пользователем и андером
    df.loc[(df['DEAL_REACHED_UNDER'] == 1) & (df['ander'] == 'Отмена') &
           (c01 | c091), 'ander'] = 'Отменено пользователем и андером'
    # Отменено только пользователем
    df.loc[(df['DEAL_REACHED_UNDER'] == 0) & (c01 | c091), 'ander'] = 'Отменено только пользователем'

    print("Всего", df.shape)
    print('Одобрено', df[df['ander'] == 'Одобрено'].shape)
    print('Отклонено', df[df['ander'] == 'Отмена'].shape)
    print('Отменено пользователем и андером', df[df['ander'] == 'Отменено пользователем и андером'].shape)
    print('Отменено пользователем, одобрено андером',
          df[df['ander'] == 'Отменено пользователем, одобрено андером'].shape)
    print('Отменено только пользователем', df[df['ander'] == 'Отменено только пользователем'].shape)
    print()
    # df[df['ander'] == 'Отмена'].to_csv('mart.csv', na_rep=' ')

    # -- target кодируем строки не требующие восполнения пропущенных
    # Все андерайтером
    # df.reset_index(inplace=True)

    # df['ander'] = df['ander'].map({'Одобрено': 1, 'Отмена': 0, 'Отменено пользователем и андером': 0,
    #                                'Отменено пользователем, одобрено андером': 1,
    #                                'Отменено только пользователем': 0, pd.NA: 0, np.nan: 0}).astype(int)

    df['ander'] = df['ander'].map({'Одобрено': 1, 'Отмена': 0, 'Отменено пользователем и андером': -1,
                                   'Отменено пользователем, одобрено андером': 1,
                                   'Отменено только пользователем': -1, pd.NA: -1, np.nan: -1}).astype(int)

    print("Всего", df.shape[0])
    print('Одобрено', df[df['ander'] == 1].shape[0])
    print('Отмена', df[df['ander'] == 0].shape[0])
    print('Отмена+Одобрено', df[df['ander'] == 0].shape[0] + df[df['ander'] == 1].shape[0])
    print()

    # df['ander'] = df['ander'].map({'Одобрено': 0, 'Отмена': 1, 'Отменено пользователем и андером': 2,
    #                                'Отменено пользователем, одобрено андером': 2,
    #                                'Отменено только пользователем': 2, pd.NA: -1, np.nan: -1}).astype(int)

    # -- удаляем лишние строки цели
    # df = df[df['DEAL_REACHED_UNDER'] == 1]
    # андером
    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    print()

    df2 = pd.read_csv('/home/u2/Desktop/a.csv')
    df2['TT'].fillna(0, inplace=True)
    df = df2.merge(df, on='APP_SRC_REF')
    df.reset_index(inplace=True, drop=True)
    print(df.head().to_string())
    df['TT'] = df['TT'].astype(int)
    print(df['TT'].unique())
    df['TT'] = pd.cut(df['TT'], 7)
    print(df['TT'])
    from sklearn.preprocessing import LabelEncoder
    print("wtf")
    le: LabelEncoder = LabelEncoder().fit(df['TT'])
    # print("wtf2", le.transform(le.classes_).shape)
    print(df['TT'].shape)
    # exit()
    df['TT'] = le.transform(df['TT'])




    df = add_scoring_resp(df)

    # -- save ids
    save('id.pickle', df['id'].tolist())
    return save('pre_process.pickle', df)


def process_by_handes(p):
    df: pd.DataFrame = pd.read_pickle(p)
    # just in case
    # df = df.reset_index(drop=True)

    # -- check unbalanced and empty columns
    if False:
        df_appr = df[df[fds] == 'Одобрено']
        df_rej = df[df[fds] == 'Отмена']
        for c in df.columns.values:
            if df_appr[c].notna().sum() == 0 or df_rej[c].notna().sum() == 0:
                print(c)
                # print(df[c].unique())
                print(df_appr.shape[0], df_appr[df_appr[c] != 0].shape[0], df_appr[c].notna().sum())
                print(df_rej.shape[0], df_rej[df_rej[c] != 0].shape[0], df_rej[c].notna().sum())

    # df.to_csv("/home/u2/evseeva/processed.csv")

    # -- Удаляем лишние столбцы
    # df = df.reset_index(drop=True)
    # c_select = ['ANKETA_SCORING',  # скоринги
    #             'OKB_SCORING',
    #             'EQUIFAX_SCORING',
    #             'FICO_SCORING',
    #             'DIGITAL_SCORING',
    #             # 'MBKI_SCORING', # не пойму как парсить
    #             #  кредитная история
    #             'OKB_RATING_SCORING',
    #             'NBKI_RATING_SCORING',
    #             'EQUIFAX_RATING_SCORING',
    #             # other
    #             'CLIENT_GENDER',
    #             'CLIENT_AGE',
    #             'CLIENT_MARITAL_STATUS',
    #             'CLIENT_DEPENDENTS_COUNT',
    #             'CLIENT_EDUCATION',
    #             'CLIENT_WI_EXPERIENCE',
    #             'PD_DATE_PUT',
    #             'INITIAL_FEE_DIV_CAR_PRICE'
    #             ]
    del_cols = [
        # 'DEAL_CONTRACT_NUMBER' # was in index
        'OPTIONAL_CONTRACT_NUMBER',
        # 'DEAL_STATUS',
        'DEAL_STATUS_NAME', 'DECISION_DEAL',  # - окончательное решение, may be used for weight
        'DEAL_KIND_CRED_NAME',
        # 'DEAL_CREATED_DATE', # время создания заявки
        'DEAL_CREATED_TIME',
        'DIFF_CREATED_RUNNING',  # время от создания, до направления на скоринг
        'DEAL_RELEASE_DATE', 'DECISION_TIME',
        'CLIENT_NAME', 'CLIENT_MOBILE_PHONE',  # can be used
        # 'CLIENT_BIRTH_DATE', # used for client age calc
        'CLIENT_ADDR_REG', 'CLIENT_ADDR_REG_REGIONKLADRID', 'CLIENT_ADDR_ACT', 'CLIENT_ADDR_ACT_REGIONKLADRID',
        # can be used.
        'CLIENT_WI_NAME', 'CLIENT_WI_INN', 'CLIENT_WI_ORG_ADDR_ACT',
        'CLIENT_WI_OKVED_NAME',
        # 'CLIENT_WI_JOB_TYPE', 'CLIENT_WI_JOB_STATUS' OPTIONAL_KASKO_NAME OPTIONAL_OSAGO_NAME OPTIONAL_OSAGO_COST # how many?
        # OPTIONAL_OSAGO_COST OPTIONAL_OSAGO_TERM OPTIONAL_OSAGO_PAYMTD # how many?
        # 'AUTO_DEAL_PDN',  # na
        # 'ANKETA_SCORING_MAX',  # nan
        # 'DEALER_NAME',  # nan
        # 'DEALER_INN',  # nan
        'DEAL_CONTRACT_DATE', 'DEAL_DATE_END',  # Дата выдачи кредита Дата погашения кредита
        'DEAL_AGE_DATE_CURRENT',  # Возраст кредита на текущую дату, мес.
        'AUTO_DEAL_EFFECTIVE_RATE', 'CAR_INFO_VIN', 'CAR_INFO_ISSUE_YEAR', 'DEAL_AGE_DATE_END', 'MARKET_CAR_PRICE',
        'PARTNER_SHOWROOM_NAME', 'PARTNER_NAME', 'MANAGER_NAME',  # can be used
        'CREDIT_INSPECTOR_NAME', 'RELEASE_MANAGER_NAME', 'OPTIONAL_KASKO_COST', 'OPTIONAL_KASKO_TERM',
        'OPTIONAL_KASKO_PAYMTD',
        # OPTIONAL_OSAGO_NAME OPTIONAL_OSAGO_COST OPTIONAL_OSAGO_TERM OPTIONAL_OSAGO_PAYMTD OPTIONAL_LIFE_NAME # how many?
        # 'OPTIONAL_OSAGO_NAME', 'OPTIONAL_LIFE_NAME',  # nan
        # OPTIONAL_WORKPLACE_NAME OPTIONAL_WORKPLACE_COST OPTIONAL_WORKPLACE_TERM OPTIONAL_WORKPLACE_TARIFF # how many?
        'OPTIONAL_WORKPLACE_NAME', 'OPTIONAL_NAME',  # nan
        # PAY_FOR_CAR_RICIPIENT AD_SUM_TOTAL_CONFIRMED AD_CAR_COST_CONFIRMED OPTIONAL_NAME OPTIONAL_COST # how many?
        # PRC_UNDER_FIO # may be used
        # SF_UNDER # -?? [1,0]
        # 'DC_REJECTED_CODES_DES',  # duble for DC_REJECTED_CODES
        'DC_CODES',  # -?? nan
        # 'DC_CODES_DES',
        'SF_SYSTEM', 'SF_SYSTEM_FUNCT',
        'SRC_SYSTEM', 'SRC_BANK',  # one value
        'DEAL_SRC_REF',
        'CLIENT_SRC_REF',
        'CLIENT_PASSPORT_SERIES', 'CLIENT_PASSPORT_NUMBER', 'CLIENT_PASSPORT_DIVISION_NAME',
        'CLIENT_PASSPORT_ISSUED_DATE',
        # 'DEAL_ID',
        # 'CLIENT_ID',
        'OPTIONAL_TELEMED_NAME',  # nan
        'DEAL_REACHED_UNDER',  # used already for target
        'FIRST_DECISION_STATE',  # used already for target
        'FIRST_DECISION_STATE_ALL',
        'SF_UNDER',  # ????
        'PRC_UNDER_REWORK_COUNT',  # Кол-во отправлений на Андеррайтинг
        # "APP_SRC_REF",  # deal_norma.src_ref
        # -- Сильная корреляция в отклоненных
        "AUTO_DEAL_SRD",
        "UNDER_NAME",
        "AD_SUM_TOTAL_CONFIRMED",
        "CAR_INFO_WEIGHT_PERMISSIBLE",
        "CAR_INFO_NUMBER_PTS",
        "CAR_INFO_IS_EPTS",
        'OPTIONAL_TELEMED_CANCEL_DATE',
        'OPTIONAL_ULTRA24_CANCEL_DATE',
        'OPTIONAL_WORKPLACE_CANCEL_DATE',
        'OPTIONAL_LIFE_CANCEL_DATE',
        'OPTIONAL_OSAGO_CANCEL_DATE',
        'OPTIONAL_KASKO_CANCEL_DATE'
    ]

    def find_date_20():
        from sklearn.model_selection import train_test_split
        target = 'ander'
        X = df.drop([target], 1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        print(X_test.head(10).to_string())

    # print(df.columns.tolist())
    # exit()
    # find_date_20()
    # df = df[c_select]  # drop other
    df.drop(columns=del_cols, inplace=True)
    dc = [c for c in df.columns if str(c).endswith("IS_CANCEL")]
    # print(dc)
    df.drop(columns=dc, inplace=True)

    # -- корректируем тип float
    float_columns = df.select_dtypes(include=["floating"]).columns
    int_col = []
    # если после точки .00 нет знаков то это INT
    for c in float_columns:
        if (df[c] - df[c].round()).sum() == 0:
            int_col.append(c)

    for c in int_col:
        # 'coerce' - if error we set NaN
        df[c] = pd.to_numeric(df[c].round(), errors='coerce').astype('Int64')

    # print(df[float_columns].tail(200).head(100).to_string())
    # df_appr = df[df[fds] == 'Одобрено']
    # df_rej = df[df[fds] == 'Отмена']

    # print(df_appr[c].describe())
    # print(df_appr[df_appr[c] == 1].shape[0])
    # print(df_rej[c].describe())
    # print(df_rej[df_rej[c] == 1].shape[0])
    # return
    # print(df.dtypes.to_string())
    # print(df.tail(10).to_string())
    # for c in df.columns:
    #     print(c, df[c].unique())
    # return

    # -- объединяем столбцы
    mms = MinMaxScaler(feature_range=(df['DIGITAL_SCORING'].min(), df['DIGITAL_SCORING'].max()))

    df['FICO_SCORING'] = mms.fit_transform(df['FICO_SCORING'].fillna(0).to_numpy().reshape(-1, 1))

    df['NBKI_SCORING'] = df['DIGITAL_SCORING'].fillna(0) + df[
        'FICO_SCORING'].fillna(0)
    df.loc[(df['DIGITAL_SCORING'].isna()) & (df['FICO_SCORING'].isna()), 'NBKI_SCORING'] = np.NAN

    df.drop(columns=['DIGITAL_SCORING'], inplace=True)
    df.drop(columns=['FICO_SCORING'], inplace=True)

    # -- создаем столбцы выделяя значимые данные
    # CAR_INFO_DATE_ISSUE_PTS
    s1 = pd.to_numeric(pd.to_datetime(df['DEAL_CREATED_AT']).dt.year).astype(
        'Int32')
    s2 = pd.to_numeric(pd.to_datetime(df['CAR_INFO_DATE_ISSUE_PTS'], errors='coerce').dt.year).astype(
        'Int32')
    df['Возраст_авто'] = s1 - s2
    df.drop(columns=['CAR_INFO_DATE_ISSUE_PTS'], inplace=True)
    # по времени
    df['Час создания заявки'] = pd.to_numeric(pd.to_datetime(df['DEAL_CREATED_AT']).dt.hour).astype(
        'Int32')
    df['Месяц создания заявки'] = pd.to_numeric(pd.to_datetime(df['DEAL_CREATED_AT']).dt.month).astype(
        'Int32')
    df['День недели'] = pd.to_numeric(pd.to_datetime(df['DEAL_CREATED_AT']).dt.dayofweek).astype(
        'Int32')

    df['CLIENT_AGE'] = pd.to_datetime(df['DEAL_CREATED_AT']).dt.year - pd.to_datetime(df['CLIENT_BIRTH_DATE']).dt.year
    df.drop(columns=['DEAL_CREATED_AT', 'CLIENT_BIRTH_DATE'], inplace=True)
    # кредитные истории
    ki_okb = 'OKB_RATING_SCORING'
    ki_nbki = 'NBKI_RATING_SCORING'
    ki_equi = 'EQUIFAX_RATING_SCORING'

    repl = {'Плохая КИ': 0,
            'Произошла ошибка': 1,
            'Произошла ошибка.': 1,
            'Ошибка выполнения': 1,
            'Ошибка выполнения проверки': 1,
            'КИ отсутствует': 2,
            'Нейтральная КИ': 3,
            'Хорошая КИ': 4
            }
    print(df['OKB_RATING_SCORING'].value_counts())

    df_c = df[[ki_okb, ki_nbki, ki_equi]].copy()
    for c in df_c.columns.values:
        print(c)
        # for p in repl:
        #     k, v = p
        #     df_c[c] = df_c[c].replace(v, k)
        # df_c[c] = df_c[c].fillna(1).astype(int)
        df_c[c] = df_c[c].map(repl)
        df_c[c] = df_c[c].fillna(1).astype(int)
    ki = {ki_okb: 'Оценка КИ ОКБ^',
          ki_nbki: 'Оценка КИ НБКИ^',
          ki_equi: 'Оценка КИ Эквифакс^'}
    df_c = df_c.rename(columns=ki)

    df = pd.concat([df, df_c], axis=1, sort=False, verify_integrity=True)

    # МБКИ
    import re
    cname = 'MBKI_SCORING'
    otchet = re.compile(r'Отчет: https:.*\n')

    def mbki_error(x: str):
        if pd.isna(x):
            return x
        else:
            if x.startswith('Couldn\'t') \
                    or x.startswith('Server error') \
                    or x.startswith('ERROR_NOT_FOUND') \
                    or x.startswith('Was not possible') \
                    or x.startswith('ERROR_') \
                    or x.startswith('Сетевой уровень') \
                    or x.startswith('cURL error') \
                    or x.startswith('Проверка на Uapi') \
                    or x.startswith('timed out') \
                    or x.startswith('timedout') \
                    or x.startswith('<') \
                    or x == '' \
                    or x.startswith('(truncated'):
                return 'Error'
            # else:
            #     m = otchet.match(x)
            #
            #     if m is not None:
            #         x = x[m.end():]
            return x

    # --

    # print MBKI_SCORING
    str_uniq_c = {}
    for c in df[cname].items():
        st = c[1]
        # if isinstance(st, str):
        sl = str(st).split('\n')
        for s in sl:
            s = str(s)
            # if 'http' not in s:
            m = re.search('http.?://[^ ]*', s)
            if m:
                sp = m.span()
                s = s[:sp[0]] + s[sp[1]:]

            if s not in str_uniq_c:
                str_uniq_c[s] = 1
            str_uniq_c[s] += 1

    z = list(str_uniq_c.items())
    k, v = list(zip(*z))
    d = pd.DataFrame({'k': k, 'v': v}).sort_values(by='v')
    print(d.to_string())
    # encode MBKI_SCORING
    df[cname] = df[cname].apply(mbki_error)  # clear errors

    df['МБКИ_треб_адрес'] = df[cname].apply(lambda x:
                                            re.search('Требуется проверка адреса регистрации/фактического',
                                                      x) is not None
                                            if pd.notna(x) else False).astype(int)
    df['МБКИ_треб_исп_пр'] = df[cname].apply(lambda x:
                                             re.search('Требуется проверка данных о наличии исполнительных про',
                                                       x) is not None
                                             if pd.notna(x) else False).astype(int)
    df['МБКИ_треб_пассп'] = df[cname].apply(lambda x:
                                            re.search('Требуется проверка информации о действительности па',
                                                      x) is not None
                                            if pd.notna(x) else False).astype(int)
    df['МБКИ_требаналотч'] = df[cname].apply(lambda x:
                                             re.search('Требуется анализ полного отчета МБКИ',
                                                       x) is not None
                                             if pd.notna(x) else False).astype(int)
    df['МБКИ_нет_огр'] = df[cname].apply(lambda x:
                                         re.search('Нет ограни',
                                                   x) is not None
                                         if pd.notna(x) else False).astype(int)
    df['МБКИ_недост'] = df[cname].apply(lambda x:
                                        re.search('Недостаточно данных для выполнения пр', x) is not None or
                                        re.search('Данные по клиенту не найд', x) is not None
                                        if pd.notna(x) else False).astype(int)
    df['МБКИ_розыск'] = df[cname].apply(lambda x:
                                        re.search('Лица, находящиеся в р',
                                                  x) is not None
                                        if pd.notna(x) else False).astype(int)
    df['МБКИ_невыполнена'] = df[cname].apply(lambda x:
                                             re.search('Проверка не выполнена',
                                                       x) is not None
                                             if pd.notna(x) else False).astype(int)
    df['МБКИ_налспецуч'] = df[cname].apply(lambda x:
                                           re.search('Наличие информации о постановке клиента на спец',
                                                     x) is not None
                                           if pd.notna(x) else False).astype(int)
    df['МБКИ_налсуд'] = df[cname].apply(lambda x:
                                        re.search('Наличие информации о судимости',
                                                  x) is not None
                                        if pd.notna(x) else False).astype(int)
    df['МБКИ_кат_и_срок'] = df[cname].apply(lambda x:
                                            re.search('Невозможно определить категорию и срок совершения преступления',
                                                      x) is not None
                                            if pd.notna(x) else False).astype(int)
    df['МБКИ_данные_не'] = df[cname].apply(lambda x:
                                           re.search('Данные по клиенту не найдены',
                                                     x) is not None
                                           if pd.notna(x) else False).astype(int)

    # for clusterization scale MBKI scoring
    # s_count = df[cname].value_counts()
    # important values count > size/15
    # unique_important = s_count[s_count > df.shape[0] / 15].shape[0]
    # for c in ['МБКИ_адрес', 'МБКИ_исп_пр', 'МБКИ_неогр', 'МБКИ_недост', 'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_спецуч', 'МБКИ_паспорт', 'МБКИ_суд', 'МБКИ_тапоМБКИ']:
    #     df[c] = df[c] / unique_important

    # return
    sc_fico = 'NBKI_SCORING'
    sc_anket = 'ANKETA_SCORING'
    sc_okb = 'OKB_SCORING'
    sc_equi = 'EQUIFAX_SCORING'
    df_scki = df[[sc_fico, sc_anket, sc_okb, sc_equi, ki[ki_okb], ki[ki_nbki], ki[ki_equi]]].copy()

    # [0-1] scale
    for c in df_scki.columns.values:
        df_scki[c].fillna(0, inplace=True)
        df_scki[c] = (df_scki[c] - df_scki[c].min()) / (df_scki[c].max() - df_scki[c].min())

    # a, b, c, d, e, f, g = 0.9, 0.9, 0.9, 0.7, 0.1, 0.1, 0.5
    # df['Синтезированный_признак_1'] = \
    #     (df_scki[ki[ki_okb]] * a +
    #      df_scki[sc_equi] * b +
    #      df_scki[ki[ki_equi]] * c +
    #      df_scki[ki[ki_nbki]] * d +
    #      df_scki[sc_okb] * e +
    #      df_scki[sc_anket] * f +
    #      df_scki[sc_fico] * g)

    # reject codes
    def rc_pos(x):
        lc = str(x).split('; \\n')
        if len(lc) >= (posit + 1):
            return lc[posit]
        else:
            return None

    posit = 0
    df['DC_REJECTED_CODES1'] = df['DC_REJECTED_CODES'].apply(rc_pos)
    posit = 1
    df['DC_REJECTED_CODES2'] = df['DC_REJECTED_CODES'].apply(rc_pos)
    posit = 3
    df['DC_REJECTED_CODES3'] = df['DC_REJECTED_CODES'].apply(rc_pos)

    def rc_count(x):
        lc = str(x).split('; \\n')
        return len(lc)

    df['DC_REJECTED_CODES_COUNT'] = df['DC_REJECTED_CODES'].apply(rc_count)

    # -- fix float for seaborn
    for c in df.select_dtypes(include="Float64").columns:
        df[c] = df[c].astype(float)

    # -- del columns 2
    del_cols = [
        'PD_DATE_PUT',  # ПД рачетный коэффициент, играет слишком сильную роль.
        'DC_REJECTED_CODES',  # заполняются андерайтором
    ]
    # df = df.drop(columns=del_cols)

    # print(df['DEAL_STATUS'].unique())
    # exit(0)

    print("Одобрено", df[df['ander'] == 0].shape[0])  # Одобрено
    print("Отклонено", df[df['ander'] == 1].shape[0])
    print("Отклонено пользователем", df[df['ander'] == 2].shape[0])

    # -- test ids
    ids: list = pd.read_pickle('id.pickle')
    print("ids check:")
    assert all(df['id'] == ids)

    return save('by_hands.pickle', df)


def model_search(p: str):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold, KFold
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)

    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    X = df.drop(['ander'], 1)
    Y = df['ander']
    # -- lienar model
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.model_selection import StratifiedKFold, KFold

    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    classifiers = [

    ]
    classifiers2 = [

        # KNeighborsClassifier(5),
        # Accuracy: 0.959092
        # AUC: 0.984420
        # SVC(kernel="linear", C=0.025),  # очень долго
        # Accuracy: 0.989507
        # AUC: 0.996897
        # SVC(gamma=2, C=1),  # слишком долго
        # GaussianProcessClassifier(1.0 * RBF(1.0)), # не хватает памяти
        DecisionTreeClassifier(max_depth=5),
        # Accuracy: 0.994318
        # AUC: 0.998924
        RandomForestClassifier(max_depth=5, n_estimators=10, ),  # max_features=1
        # Accuracy: 0.954724
        # AUC: 0.997325
        MLPClassifier(alpha=1, max_iter=1000),
        # Accuracy: 0.987554
        # AUC: 0.985448
        AdaBoostClassifier(),
        # Accuracy: 0.994389
        # AUC: 0.999528
        GaussianNB(),
        # Accuracy: 0.986630
        # AUC: 0.995478
        QuadraticDiscriminantAnalysis()]
    # Accuracy: 0.421153
    # AUC: 0.956211

    kfold = StratifiedKFold(n_splits=4)

    # est = linear_model.RidgeClassifierCV()
    # best LogisticRegressionCV
    # Accuracy: 0.990228
    # AUC: 0.998134
    # without scaller
    # Accuracy: 0.990266
    # AUC: 0.998136
    # with scaller
    # Accuracy: 0.988486
    # AUC: 0.997062
    def _check_model(est, X, Y):
        results = cross_validate(est, X, Y, cv=kfold, scoring=['accuracy', 'roc_auc'])
        print(est.__class__.__name__)
        print("Accuracy: %f" % results['test_accuracy'].mean())
        print("AUC: %f" % results['test_roc_auc'].mean())

    # _check_model(linear_model.LogisticRegressionCV(max_iter=30), X, Y)
    # Accuracy: 0.972143
    # AUC: 0.994951
    # _check_model(linear_model.SGDClassifier(), X, Y)

    # for c in classifiers:
    #     _check_model(c, X, Y)

    _check_model(linear_model.LogisticRegressionCV(), X, Y)
    # Accuracy: 0.987802
    # AUC: 0.996564


def linear_analysis(p: str):
    from matplotlib import pyplot as plt
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    # df['ander'] = df['ander'].map({'Одобрено': 0, 'Отмена': 1, 'user_rej': 2}).astype(int)
    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    X = df.drop(['ander'], 1)
    Y = df['ander']
    # --
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold, KFold
    # -- check model
    kfold = StratifiedKFold(n_splits=4)
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    est = linear_model.RidgeCV()  # LogisticRegressionCV(solver="sag", max_iter=10)
    results = cross_val_score(est, X, Y, cv=kfold)
    print("Accuracy: %f" % results.mean())
    # -- fit all
    est.fit(X, Y)
    # -- permutation importance
    from sklearn.inspection import permutation_importance
    results = permutation_importance(est, X, Y, n_repeats=1)
    w = results.importances_mean
    cdf = pd.DataFrame(w, columns=['weight'])
    cdf['column'] = X.columns
    cdf_p = cdf.sort_values(by=['weight'], ascending=False)
    cdf_p = cdf_p.set_index('column')
    print(cdf_p.head(10))
    # return

    # print(cdf_p.to_string(), '\n')
    cdf_p.head(10).plot.barh()
    plt.subplots_adjust(left=0.61)
    plt.show()

    # -- syntesize and check
    X['new'] = 0
    cdfh = cdf_p.head(6)
    for i, c in enumerate(cdfh.index.values):
        if i == 0:
            continue

        v = cdfh.loc[c]['weight']
        X['new'] += X[c] * v
    # -- check new column
    results = cross_val_score(est, pd.DataFrame(X['sum_43']).astype(float), Y, cv=kfold)
    print("Accuracy sum_43: %f" % results.mean())
    results = cross_val_score(est, pd.DataFrame(X['new']).astype(float), Y, cv=kfold)
    print("Accuracy new feature: %f" % results.mean())


def plot_x_linear(p):
    from sklearn import linear_model
    from scipy.special import expit
    from matplotlib import pyplot as plt
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    # df['ander'] = df['ander'].map({'Одобрено': 0, 'Отмена': 1, 'user_rej': 2}).astype(int)
    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    X = df.drop(['ander'], 1)
    y = df['ander']
    c_interest = 'AUTO_DEAL_SRD'
    pos = X.columns.to_list().index(c_interest)
    # --
    clf = linear_model.LogisticRegression(C=1000)
    print(X.shape)
    # return
    # X = X[c_interest].ravel() #.reshape(56322,1)
    clf.fit(X[c_interest].ravel().reshape(56322, 1), y)
    x_test = X.tail(20)[c_interest]
    y_test = y.tail(20)
    print(clf.coef_)
    loss = expit(x_test * clf.coef_[0] + clf.intercept_).ravel()
    print(loss.shape)
    # return

    plt.scatter(x_test, y_test)
    from sklearn import linear_model
    plt.plot(x_test, loss)
    plt.show()


def feature_importance_forest2(p: str, target: str):
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
    params = {'n_estimators': range(90, 100, 1), 'min_samples_split': [2],  # 'max_leaf_nodes': range(250, 450, 10),
              'max_depth': range(4, 6, 1),
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
    est = RandomForestClassifier(class_weight={0: 1, 1: 0.75}, max_depth=4,
                                 max_features='sqrt', random_state=25)

    def my_scoring_func(y_true, y_pred):
        p = metrics.precision_score(y_true, y_pred)
        a = metrics.accuracy_score(y_true, y_pred)
        return (a * 2.5 + p) / 2

    score = make_scorer(my_scoring_func, greater_is_better=True)

    # clf = RandomizedSearchCV(RandomForestClassifier(), params, cv=kfold, scoring=score,  # scoring='roc_auc',
    #                          n_jobs=4, verbose=2, n_iter=n_iter)

    importance_sum = np.zeros(X.shape[1], dtype=float)
    c = 0
    for i in range(3, 5):
        for j in range(3, 5):
            # it is faster than cross validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True,
                                                                random_state=i + j * 3)
            c += 1
            print(c)
            kfold = StratifiedKFold(n_splits=i, shuffle=True)
            gs = RandomizedSearchCV(est, params, cv=kfold, n_iter=20, n_jobs=4, random_state=i + j * 3,
                                    # scoring=score
                                    )
            results = gs.fit(X_train, y_train)
            model = results.best_estimator_
            print(model)
            # score
            y_score = model.predict_proba(X_test)[:, 1]
            auc = metrics.roc_auc_score(y_test, y_score)
            sc = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            pes = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)

            print("Accuracy: %f" % sc)
            print("AUC: %f" % auc)
            print("Precision: %f" % pes)
            print("Recall: %f" % recall)

            # FEATURE IMPORTANCE
            imp = model.feature_importances_  # feature importance
            # scale
            imp = np.reshape(imp, [-1, 1])
            imp = preprocessing.MinMaxScaler().fit_transform(imp)
            imp = np.ravel(imp)

            importance_sum += imp

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking forest:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importance_sum[indices[f]] / 100))
    return importance_sum, c


def feature_importance_xgboost_static(p: str, target: str, perm: bool = False, nrep=5):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop(columns=[target, 'id'])
    y = df[target]

    # X.drop(columns=["index", ""], inplace=True)

    X_columns = X.columns

    X = StandardScaler().fit_transform(X)  # XGB specific

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, shuffle=True, random_state=i)
    # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                       colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #                       gamma=2, gpu_id=0, importance_type='gain',
    #                       interaction_constraints='', learning_rate=0.300000012,
    #                       max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
    #                       monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
    #                       num_parallel_tree=1, random_state=i, reg_alpha=0.2, reg_lambda=1,  # 22
    #                       scale_pos_weight=0.34263295553618134, seed=22, subsample=1,
    #                       tree_method='exact', use_label_encoder=False,
    #                       validate_parameters=1, verbosity=1)
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                          colsample_bynode=1, colsample_bytree=0.8, eval_metric='logloss',
                          gamma=2, gpu_id=0, importance_type='gain',
                          interaction_constraints='', learning_rate=0.300000012,
                          max_delta_step=0, max_depth=2, min_child_weight=6, missing=np.nan,
                          monotone_constraints='()', n_estimators=140, n_jobs=2, nthread=2,
                          num_parallel_tree=1, random_state=7, reg_alpha=0.2, reg_lambda=1,
                          scale_pos_weight=0.28, seed=7, subsample=1, tree_method='exact',  # 0.7 - with otcl
                          use_label_encoder=False, validate_parameters=1, verbosity=1)
    # Accuracy: 0.704202
    # AUC: 0.714913
    # Precision: 0.791685
    # Recall: 0.015535

    # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
    #                   importance_type='gain', interaction_constraints='',
    #                   learning_rate=0.02, max_delta_step=0, max_depth=6,
    #                   min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #                   n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
    #                   random_state=42, reg_alpha=0.5, reg_lambda=1,
    #                   scale_pos_weight=0.75, seed=42, subsample=1,  # 1.3346062351320898
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
    # Accuracy: 0.712278
    # AUC: 0.729373
    # Precision: 0.665635
    # Recall: 0.128473

    model.fit(X, y)
    # -- score
    # check_model_sklearn(model, X, y)

    # y_score = model.predict_proba(X_test)[:, 1]
    # auc = metrics.roc_auc_score(y_test, y_score)
    # sc = model.score(X_test, y_test)
    # y_pred = model.predict(X_test)
    # pes = metrics.precision_score(y_test, y_pred)
    #
    # print("Accuracy: %f" % sc)
    # print("AUC: %f" % auc)
    # print("Precision: %f" % pes)

    # FEATURE IMPORTANCE
    if perm:
        imp = permutation_importance(model, X, y, n_repeats=nrep).importances_mean
    else:
        imp = model.feature_importances_

    # scale
    imp = np.reshape(imp, [-1, 1])
    imp = preprocessing.MinMaxScaler().fit_transform(imp)
    importance_sum = np.ravel(imp)

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking xgboost:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X_columns[indices[f]], importance_sum[indices[f]] / 100))

    return importance_sum, 0


def feature_importance_xgboost_perm_binary(p: str, target: str, perm: bool = False, nrep=5) -> (np.ndarray, int):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop(columns=[target, 'id'])
    y = df[target]

    # y = LabelBinarizer().fit_transform(y) # for multiclass

    # print(np.unique(np.asarray(y)))
    # exit()

    X_columns = X.columns

    X = StandardScaler().fit_transform(X)  # XGB specific

    importance_sum = np.zeros(X.shape[1], dtype=float)
    c = 0
    for i in range(1, 2):
        c += 1
        print(c)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, shuffle=True, random_state=i)
        # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                       colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
        #                       gamma=2, gpu_id=0, importance_type='gain',
        #                       interaction_constraints='', learning_rate=0.300000012,
        #                       max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
        #                       monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
        #                       num_parallel_tree=1, random_state=i, reg_alpha=0.2, reg_lambda=1,  # 22
        #                       scale_pos_weight=0.34263295553618134, seed=22, subsample=1,
        #                       tree_method='exact', use_label_encoder=False,
        #                       validate_parameters=1, verbosity=1)
        # Accuracy: 0.704202
        # AUC: 0.714913
        # Precision: 0.791685
        # Recall: 0.015535

        model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
                              importance_type='gain', interaction_constraints='',
                              learning_rate=0.02, max_delta_step=0, max_depth=6,
                              min_child_weight=1, missing=np.nan, monotone_constraints='()',
                              n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
                              random_state=i, reg_alpha=0.5, reg_lambda=1,
                              seed=i*10, subsample=1,  # 1.3346062351320898 # scale_pos_weight=0.75,
                              tree_method='exact', use_label_encoder=False,
                              validate_parameters=1, verbosity=1)
        # Accuracy: 0.712278
        # AUC: 0.729373
        # Precision: 0.665635
        # Recall: 0.128473
        #
        # model = XGBClassifier(base_score=0.3, booster='gbtree', colsample_bylevel=1,
        #                   colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
        #                   eval_metric='logloss', gamma=0.1, gpu_id=0, importance_type=None,
        #                   interaction_constraints='', learning_rate=0.03, max_delta_step=0,
        #                   max_depth=20, min_child_weight=1, missing=np.nan,
        #                   monotone_constraints='()', n_estimators=152, n_jobs=2, nthread=4,
        #                   num_parallel_tree=1, predictor='auto', random_state=41,
        #                   reg_alpha=0.6, reg_lambda=1, scale_pos_weight=0.28, seed=41,
        #                   subsample=0.3, tree_method='exact', use_label_encoder=False)

        model.fit(X, y)
        # print(model.objective)
        # print(model.n_classes)
        # exit()
        # -- score
        # check_model_sklearn(model, X, y)

        # y_score = model.predict_proba(X_test)[:, 1]
        # auc = metrics.roc_auc_score(y_test, y_score)
        # sc = model.score(X_test, y_test)
        # y_pred = model.predict(X_test)
        # pes = metrics.precision_score(y_test, y_pred)
        #
        # print("Accuracy: %f" % sc)
        # print("AUC: %f" % auc)
        # print("Precision: %f" % pes)

        # FEATURE IMPORTANCE
        if perm:
            imp = permutation_importance(model, X, y, n_repeats=nrep).importances_mean
        else:
            imp = model.feature_importances_

        # scale
        imp = np.reshape(imp, [-1, 1])
        imp = preprocessing.MinMaxScaler().fit_transform(imp)
        imp = np.ravel(imp)

        importance_sum += imp

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking xgboost:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X_columns[indices[f]], importance_sum[indices[f]] / 100))

    return importance_sum, c


def feature_importance_xgboost_perm_multi(p: str, target: str, perm: bool = False, nrep=5) -> (np.ndarray, int):
    """ return importance, count of modes was tested"""

    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop(columns=[target, 'id'])
    y = df[target]

    # y = LabelBinarizer().fit_transform(y) # for multiclass

    # print(np.unique(np.asarray(y)))
    # exit()

    X_columns = X.columns

    X = StandardScaler().fit_transform(X)  # XGB specific

    importance_sum = np.zeros(X.shape[1], dtype=float)
    c = 0
    for i in range(1, 2):
        c += 1
        print(c)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, shuffle=True, random_state=i)
        # model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                       colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
        #                       gamma=2, gpu_id=0, importance_type='gain',
        #                       interaction_constraints='', learning_rate=0.300000012,
        #                       max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
        #                       monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
        #                       num_parallel_tree=1, random_state=i, reg_alpha=0.2, reg_lambda=1,  # 22
        #                       scale_pos_weight=0.34263295553618134, seed=22, subsample=1,
        #                       tree_method='exact', use_label_encoder=False,
        #                       validate_parameters=1, verbosity=1)
        # Accuracy: 0.704202
        # AUC: 0.714913
        # Precision: 0.791685
        # Recall: 0.015535

        model = XGBClassifier(base_score=0.5, booster='gbtree',  colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
                              importance_type='gain', interaction_constraints='',
                              learning_rate=0.02, max_delta_step=0, max_depth=6,
                              min_child_weight=1, missing=np.nan, monotone_constraints='()',
                              n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
                              random_state=i, reg_alpha=0.5, reg_lambda=1,
                              seed=i * 10, subsample=1,  # 1.3346062351320898 # scale_pos_weight=0.75,
                              tree_method='exact', use_label_encoder=False,
                              validate_parameters=1, verbosity=1)
        # Accuracy: 0.712278
        # AUC: 0.729373
        # Precision: 0.665635
        # Recall: 0.128473
        #
        # model = XGBClassifier(base_score=0.3, booster='gbtree', colsample_bylevel=1,
        #                   colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
        #                   eval_metric='logloss', gamma=0.1, gpu_id=0, importance_type=None,
        #                   interaction_constraints='', learning_rate=0.03, max_delta_step=0,
        #                   max_depth=20, min_child_weight=1, missing=np.nan,
        #                   monotone_constraints='()', n_estimators=152, n_jobs=2, nthread=4,
        #                   num_parallel_tree=1, predictor='auto', random_state=41,
        #                   reg_alpha=0.6, reg_lambda=1, scale_pos_weight=0.28, seed=41,
        #                   subsample=0.3, tree_method='exact', use_label_encoder=False)

        model.fit(X, y)
        # print(model.objective)
        # print(model.n_classes)
        # exit()
        # -- score
        # check_model_sklearn(model, X, y)

        # y_score = model.predict_proba(X_test)[:, 1]
        # auc = metrics.roc_auc_score(y_test, y_score)
        # sc = model.score(X_test, y_test)
        # y_pred = model.predict(X_test)
        # pes = metrics.precision_score(y_test, y_pred)
        #
        # print("Accuracy: %f" % sc)
        # print("AUC: %f" % auc)
        # print("Precision: %f" % pes)

        # FEATURE IMPORTANCE
        if perm:
            imp = permutation_importance(model, X, y, n_repeats=nrep).importances_mean
        else:
            imp = model.feature_importances_

        # scale
        imp = np.reshape(imp, [-1, 1])
        imp = preprocessing.MinMaxScaler().fit_transform(imp)
        imp = np.ravel(imp)

        importance_sum += imp

    indices = np.argsort(importance_sum)[::-1]  # sort indexes

    # Print the feature ranking
    print("Feature ranking xgboost:")
    print(importance_sum.shape)

    for f in range(X.shape[1])[:100]:  # первые 100
        print("%d. %s (%f)" % (f + 1, X_columns[indices[f]], importance_sum[indices[f]] / 100))

    return importance_sum, c


def feature_importance_composite(p: str, target: str):
    """TODO add weight by AUC"""
    from sklearn import preprocessing

    df: pd.DataFrame = pd.read_pickle(p)
    # print(df.isna().sum().to_string())
    # print(df.dtypes.to_string())
    # print(df.shape)
    # print(df.columns.tolist())
    # exit()

    X = df.drop(columns=[target, 'id'])
    X_columns = X.columns

    im1, count = feature_importance_forest2(p, target)
    im2, count = feature_importance_xgboost(p, target)

    im1 = np.reshape(im1, [-1, 1])
    im1 = preprocessing.MinMaxScaler().fit_transform(im1)
    im1 = np.ravel(im1)
    im2 = np.reshape(im2, [-1, 1])
    im2 = preprocessing.MinMaxScaler().fit_transform(im2)
    im2 = np.ravel(im2)
    return save('feature_importance_comp.pickle', (X_columns, im1, im2))


def feature_importance_composite_permut(p: str, target: str, nrep_forst=2, nrep_xgboost=5):
    """TODO add weight by AUC"""
    from sklearn import preprocessing

    df: pd.DataFrame = pd.read_pickle(p)
    # print(df.isna().sum().to_string())
    # print(df.dtypes.to_string())
    # print(df.shape)
    # print(df.columns.tolist())
    # exit()

    X = df.drop(columns=[target, 'id'])
    X_columns = X.columns

    # im1, count = feature_importance_forest2(p, target)
    # im1, count = permutation_importance_forest(p, target, nrep=nrep_forst)
    im2, count = feature_importance_xgboost_perm_binary(p, target, perm=True, nrep=nrep_xgboost)
    im2, count = feature_importance_xgboost_perm_multi(p, target, perm=True, nrep=nrep_xgboost)


    im1 = np.reshape(im1, [-1, 1])
    im1 = preprocessing.MinMaxScaler().fit_transform(im1)
    im1 = np.ravel(im1)
    im2 = np.reshape(im2, [-1, 1])
    im2 = preprocessing.MinMaxScaler().fit_transform(im2)
    im2 = np.ravel(im2)
    return save('feature_importance_comp.pickle', (X_columns, im1, im2))


def feature_importance_composite_plot(p: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    X_columns, im1, im2 = pd.read_pickle(p)
    for k in X_columns:
        if k not in fields_dict:
            fields_dict[k] = k
    #  ----- RENAME COLUMNS: -----
    # X_columns = X_columns.map(fields_dict)

    print(X_columns)
    # -- отдельно
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Важность признаков для XGBoost и RandomForest")
    df_imp: pd.DataFrame = pd.DataFrame({"RandomForest": im1, "XGBoost": im2}, index=X_columns)
    df_imp.sort_values(by=["XGBoost"], ascending=False, inplace=True)
    df_imp = df_imp.head(20)
    df_imp.plot.bar(rot=0, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    fig.subplots_adjust(bottom=0.638)
    plt.savefig("Важность признаков для XGBoost и RandomForest")
    # plt.show()

    # -- Сумма
    im1 = np.array(im1) * 0.97 * 3  # accuracy
    im2 = np.array(im2) * 0.99 * 3  # accuracy

    importance_sum = im1 + im2

    df_imp = pd.DataFrame({"names": X_columns, "importance": importance_sum})
    df_imp.sort_values(by=["importance"], ascending=False, inplace=True)
    print('Важность признаков')
    for i, r in enumerate(df_imp.iterrows()):
        print("%d. %s (%f)" % (i + 1, r[1][0], r[1][1]))

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.36)

    df_imp = df_imp.head(25)
    df_imp['importance'] = df_imp['importance']  # np.sqrt
    sns.barplot(x=df_imp['importance'], y=df_imp['names'], ax=ax)
    fig.suptitle("Важность признаков сумма XGBoost и RandomForest")
    fig.savefig("feature_importance")
    # plt.show()


def pair_scatter_plot(p):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    df: pd.DataFrame = pd.read_pickle(p)
    print(df)
    # print(df.dtypes.to_string())
    # print(df['CAR_INFO_NUMBER_PTS'].unique())
    # return

    # -- select columns

    # df = df[[
    #     'CAR_INFO_IS_EPTS',
    #     'sum_43',
    #     'EQUIFAX_SCORING',
    #     'Оценка КИ ОКБ^',
    #     'OKB_RATING_SCORING_КИ отсутствует',
    #     'OPTIONAL_ULTRA24_COST',
    #     'OKB_RATING_SCORING_Хорошая КИ',
    #     'OKB_SCORING',
    #     'ander',
    # ]]
    # -- rename columns
    # df = rename_columns(df, fields_dict, 15)
    print(df.columns)
    # -- plot
    # plot_kws={"s":3}
    kind = "hist"
    sns.pairplot(df, kind=kind, hue='ander',
                 diag_kind='hist')  # , hue='ander', ) # palette=["black", "red"], plot_kws=dict(s=50, alpha=0.8), markers=["^", "v"]
    # plt.show()
    plt.savefig("pair_scatter_plot_feature_importance_" + kind)


def test_model_own(p1, p2):
    target = 'ander'
    # -- data
    df_train: pd.DataFrame = pd.read_pickle(p1)
    df_test: pd.DataFrame = pd.read_pickle(p2)

    print(df_train.columns)
    print("Всего train:", df_train.shape)
    print("Отклонено", df_train[df_train[target] == 0].shape)
    print("Одобрено", df_train[df_train[target] == 1].shape)
    print()

    print("Всего test:", df_test.shape)
    print("Отклонено", df_test[df_test[target] == 0].shape)
    print("Одобрено", df_test[df_test[target] == 1].shape)
    print()

    # X_train: pd.DataFrame = df_train.drop(columns=[target, 'id'])
    # X_test: pd.DataFrame = df_test.drop(columns=[target, 'id'])
    X_train: pd.DataFrame = df_train.drop(columns=[target, 'id'])
    X_test: pd.DataFrame = df_test.drop(columns=[target, 'id'])
    X_test['id'] = df_test['id']
    y_train = df_train[target]
    y_test = df_test[target]

    from estimator_xgboost_based import MyEstimXGB, MyEstimXGB_wz

    # m = MyEstimXGB(v1a=-0.45, v1b=0.1, v1c=0.05, v2a=0.6, v2b=255, v2c=2.2,
    #                v3a=80, v3b=-0.3, v3c=0.1, v4a=-0.15, v4b=0.02, v5a=0, v5b=0, v5c=0)
    # {'v1a': -0.325, 'v1b': 0.1, 'v1c': 0.15, 'v2a': 0.55, 'v2b': 245, 'v2c': 2.25, 'v3a': 70, 'v3b': -0.31,
    #  'v3c': 0.375, 'v4a': -0.15, 'v4b': 0.0, 'v5a': 0.24285714285714285, 'v5b': -0.2, 'v5c': 0.26}
    # m = MyEstimXGB(v1a=-0.325, v1b=0.1, v1c=0.15, v2a=0.55, v2b=245, v2c=2.25,
    #                v3a=70, v3b=-0.31, v3c=0.375, v4a=-0.15, v4b=0, v5a=0.24, v5b=-0.2, v5c=0.26)
    # m = MyEstimXGB(v1a=-0.325, v1b=0.2, v1c=0.0, v2a=0.65, v2b=240.0, v2c=2.3,
    #                       v3a=63.75, v3b=-0.05, v3c=0.2, v4a=-0.25, v4b=0.1, v5a=0.25,
    #                       v5b=-0.2, v5c=0.36)
    # m = MyEstimXGB(v1a=-0.32, v1b=0.18, v1c=0.02, v2a=0.0026333333333333334, v2b=2.4,
    #        v3a=61.666666666666664, v3b=-0.04, v3c=0.18333333333333332,
    #        v4a=-0.35, v4b=1.3877787807814457e-17, v5a=0.21666666666666667,
    #        v5b=-0.16333333333333333, v5c=0.35333333333333333,
    #        v6a=0.0010999999999999998, v6b=0.835, v6c=-0.1225, v7=0.4, v71a=0.1,
    #        v71b=-0.075, v72a=0.2, v72b=-0.03, v73a=-0.01, v73b=0.05)
    # m = MyEstimXGB(v1a=-0.32, v1b=0.19, v1c=0.02, v2a=0.0028, v2b=2.4,
    #                       v3a=61.666666666666664, v3b=-0.03, v3c=0.21666666666666667,
    #                       v4a=-0.2833333333333333, v4b=0.05, v5a=0.21666666666666667,
    #                       v5b=-0.19, v5c=0.3666666666666667, v6a=0.00105, v6b=0.83, v6c=-0.1,
    #                       v7=1.0, v71b=-0.2, v72b=-0.055, v73b=0.05)
    # m = MyEstimXGB(v1a=-0.32, v1b=0.175, v1c=0.02333333333333333,
    #            v2a=0.0028666666666666667, v2b=2.466666666666667, v3a=61.0,
    #            v3b=-0.02, v3c=0.23, v4a=-0.26, v4b=0.03266666666666667,
    #            v5a=0.21333333333333332, v5b=-0.2, v5c=0.36,
    #            v6a=0.0010333333333333334, v6b=0.84, v6c=-0.06666666666666667,
    #            v7a=0.9, v71b=-0.18, v72b=-0.07, v73b=0.1)
    # --- BEST ONE!!! -----
    # m = MyEstimXGB_wz(v1a=-1, v1b=+1, v1c=+1, v2a=0.0027, v2b=2.27, v3a=+1,
    #                v3b=+1, v3c=+1, v4a=+1, v4b=+1, v5a=+1, v5b=+1,
    #                v5c=-1, v6a=0.001, v6b=0.811, v6c=+1, v71b=+1, v72b=-1,
    #                v73b=-1, v7a=+1, v7b=-1)
    # --- WHITE ZOME
    # m = MyEstimXGB_wz(v1a=-0.32, v1b=0.05686666666666662, v1c=0.0010000000000000009,
    #           v2a=0.00271, v2b=2.271, v3a=63.5, v3b=0.02600000000000001,
    #           v3c=0.010000000000000009, v4a=0.0010000000000000009, v4b=0.051,
    #           v5a=0.21, v5b=0.201, v5c=-0.11266666666666666, v6a=0.001,
    #           v6b=0.0014999999999999458, v6c=0.12000000000000005, v71b=-0.24,
    #           v72b=-0.053, v73b=-0.143, v7a=0.92, v7b=0.07333333333333333)

    m = MyEstimXGB_wz(v1a=-1, v1b=1, v1c=1, v2a=0.0027322222222222227,
                  v2b=2.236111111111111, v3a=1, v3b=1, v3c=1, v4a=1, v4b=1, v5a=1,
                  v5b=1, v5c=-1, v6a=0.009772222222222222, v6b=0.5217777777777778,
                  v6c=1, v71b=1, v72b=-1, v73b=-1, v7a=1, v7b=-1)
    # m.set_params()  # required for MyEstimXGB
    splits = 5
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
    #                                                     random_state=1, stratify=None)

    # # explore
    # from sklearn.model_selection import cross_val_predict
    # kfold = StratifiedKFold(n_splits=splits)
    # y_pred = cross_val_predict(m, X_test, y_test, cv=kfold)
    # print(y_pred)
    # i_cl0 = np.where(y_pred == 1)[0]
    #
    # m2 = MyEstimXGB(v1a=-0.45, v1b=0.1, v1c=0.05, v2a=0.6, v2b=255, v2c=2.2,
    #                v3a=80, v3b=-0.3, v3c=0.1, v4a=-0.15, v4b=0.02, v5a=0, v5b=0, v5c=0)
    # y_pred2 = cross_val_predict(m2, X_test, y_test, cv=kfold)
    # i_cl02 = np.where(y_pred2 == 1)[0]
    # # ind = df.index[i_cl02[102]]
    # # print(ind)
    # # print(df.loc[ind])
    # # print()
    # # print(df_origin.loc[ind].to_string())
    # i_inter = np.intersect1d(i_cl0, i_cl02)
    # # print(len(i_cl0), len(i_cl02))
    # # print(len(i_inter))
    #
    # i_inter_ind = X_test.index[i_inter]
    # print(i_inter_ind)
    # df_origin = df_origin.iloc[i_inter_ind]
    # # print(df_origin[df_origin['DEAL_STATUS'] == 'release'].shape)
    # print(df_origin.loc[i_inter_ind]['APP_SRC_REF'].to_list())
    # return
    # # y = y_test.loc[i_inter_ind]
    # df = pd.concat([X_test, y_test], axis=0)
    #
    #
    # return
    # l = df.columns.to_list()
    # l[0] = 'ander'
    # df.columns = l
    # df = pd.concat([df, df_origin.iloc[200, 201]], axis=1)
    # print(df.columns[0])
    # print(df.columns)
    # return
    # save('tmp.pickle', X_test.loc[i_inter_ind])
    #
    # frequency_analysis('tmp.pickle')
    #
    # exit(0)
    check_model_sklearn_cross(m, X_train, y_train, splits=splits)

    print("-----")
    train_rate = sum(y_train) / (len(y_train) - sum(y_train))
    test_rate = sum(y_test) / (len(y_test) - sum(y_test))
    rate = (train_rate + test_rate) / 2
    print("train_rate", train_rate, "test_rate", test_rate, "rate", rate)
    X_test, y_test = downsample(X_test, y_test, rate=rate)
    test_rate = sum(y_test) / (len(y_test) - sum(y_test))
    print("test_rate after", test_rate)
    print()
    print("Всего test:", df_test.shape)
    print("Отклонено", df_test[df_test[target] == 0].shape)
    print("Одобрено", df_test[df_test[target] == 1].shape)

    check_model_sklearn_split(m, X_train, y_train, X_test, y_test, calc_confusion_matrix=True)
    print("одоб", sum(y_test))
    print("отклоненных", len(y_test) - sum(y_test))

    exit()
    m.fit(X_train, y_train)  # -- train
    from myown_pack.plot import histogram_two_in_one
    y_pred = m.pred(X_test)
    print(y_pred)
    print("np.max(y_pred[:, 1])", np.max(y_pred[:, 1]))
    positive = y_pred[:, 1]/np.max(y_pred[:, 1])
    negative = y_pred[:, 0]/np.max(y_pred[:, 1])
    print(positive)
    # df = pd.DataFrame({'INITIAL_FEE_DIV_CAR_PRICE': X_test['INITIAL_FEE_DIV_CAR_PRICE'],
    #                    'ander': y_test, 'positive': positive, 'negative': negative})
    exit()
    df = X_test
    df['ander'] = y_test
    df['positive'] = positive
    df['negative'] = negative
    df['p-n'] = df['positive'] - df['negative']
    print("wtf")
    # for c in X_test.columns:
    df['ACCEPTED_BY_MODE'] = df['p-n'] > 0
    df['ACCEPTED_BY_MODE'] = df['ACCEPTED_BY_MODE'].astype(int)
    # df_acc = df[df['ACCEPTED_BY_MODE'] > 0]
    # df_acc = df_acc[df_acc['ander'] == 0]
    #
    # df_acc = df[(df['ACCEPTED_BY_MODE'] > 0) and (df['ander'] > 0)]
    # print(df_acc['AUTO_DEAL_COST_REQUESTED'].sum())
    # exit()
    # df = rename_columns(df, columns=fields_dict, len=99)

    # df.to_csv('test_10m.csv')
    # p = 'by_hands.pickle'
    p = 'select_test.pickle'
    df_o: pd.DataFrame = pd.read_pickle(p)
    df_o = df_o.set_index("id",drop=True)
    # df_se = df['id', 'ACCEPTED_BY_MODE', 'positive', 'negative', 'p-n']
    print(df.shape, X_test.shape)
    df_all: pd.DataFrame = df_o.loc[X_test["id"].astype(int).to_list()]
    df = df.set_index("id", drop=True)
    for c in ['ACCEPTED_BY_MODE', 'positive', 'negative', 'p-n']:
        df_all[c] = df[c]
    df_all.to_csv('test_10m.csv')
    exit()
    # replace_names(fields_dict=fields_dict)

    # import seaborn as sns
    # # df.plot.scatter(y=c, x='positive', c='ander', colormap='PiYG', alpha=0.5)
    #
    # for c in df.columns:
    #     print(c)
    #     if c in ['OKB_SCORING', 'МБКИ_треб_пассп', 'МБКИ_невыполнена']:
    #         break
    #     sns.scatterplot(data=df, y=c, x='positive', hue='ander')
    #     sns.kdeplot(y=df[c], x=df['positive'], hue=df['ander'])
    #     # df.plot.scatter(y=c, x='positive', c='ander', colormap='PiYG', alpha=0.5)
    #     plt.savefig(f'scat_{c}.png')
    #     plt.close()
    # c = 'AUTO_DEAL_COST_REQUESTED'  # 'INITIAL_FEE_DIV_CAR_PRICE'
    # sns.scatterplot(data=df, y=c, x='positive', hue='ander')
    # sns.kdeplot(y=df[c], x=df['positive'], hue=df['ander'])
    # plt.savefig(f'scat_{c}.png')
    # plt.close()

    # print(df)

    # print(df)
    # histogram_two_in_one(df=df, feature_main='p-n', feature_binary='ander', bins=20)
    histogram_two_in_one(df=df, feature_main='positive', feature_binary='ander', bins=20, density=False)
    return m


def test_model_own_dtree_cases(p):
    from sklearn.base import BaseEstimator
    from sklearn import metrics
    target = 'ander'
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    print(df.columns)
    # return
    X: pd.DataFrame = df.drop([target], 1)
    y = df[target]

    def log_dtree(X: pd.DataFrame, cl=(1, 2, 3, 4, 5, 6, 7)):
        preds = []
        for i, r in X.iterrows():
            res = 0
            if 1 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] > 745.5 and \
                        r['CLIENT_WI_EXPERIENCE'] < 66.5 and \
                        r['CLIENT_MARITAL_STATUS'] == 0 and \
                        r['ANKETA_SCORING'] > 90.5 and \
                        r['AUTO_DEAL_INITIAL_FEE'] <= 40500:
                    res = 1
            if 2 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] > 808.5 and \
                        r['CLIENT_WI_EXPERIENCE'] > 197.5:
                    res = 1
            if 3 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] > 808.5:
                    if r['CLIENT_WI_EXPERIENCE'] <= 197.5 and \
                            r['EQUIFAX_SCORING'] > 922:  # WHITE both
                        res = 1
            if 4 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] > 808.5:
                    if r['CLIENT_WI_EXPERIENCE'] > 197.5:  # WHITE
                        if r['AUTO_DEAL_INITIAL_FEE'] > 760150:  # WHITE - more precisely
                            res = 1
            if 5 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] <= 745.5 and r['AUTO_DEAL_INITIAL_FEE'] <= 44500:
                    if 500 < r['AUTO_DEAL_INITIAL_FEE'] <= 22500:  # white
                        res = 1
            if 6 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        846.5 >= r['EQUIFAX_SCORING'] > 745.5 and r['AUTO_DEAL_INITIAL_FEE'] <= 48500:
                    res = 1
            if 7 in cl:
                if r['OKB_RATING_SCORING_КИ отсутствует'] == 0 and \
                        r['EQUIFAX_SCORING'] > 846.5 and \
                        r['CLIENT_MARITAL_STATUS'] == 0 and r['OKB_SCORING'] <= 810:
                    res = 1

            preds.append(res > 0)
        return preds

    # -- estimator
    class MyEstim(BaseEstimator):
        def __init__(self, pred: callable, cl: list):
            self.pred = pred
            self.cl = cl

        def fit(self, X=None, y=None):
            pass

        def predict(self, X):
            return self.pred(X, cl=self.cl)

        def score(self, X, y):
            y_pred = self.pred(X, cl=self.cl)
            return metrics.accuracy_score(y, y_pred)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True,
    #                                                     random_state=1, stratify=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
                                                        random_state=1, stratify=None)
    odobs = []
    scs = []
    pess = []
    recalls = []
    for i in range(1, 8):
        m = MyEstim(pred=log_dtree, cl=[i])
        m.fit(X_train, y_train)
        # test
        sc = m.score(X_test, y_test)
        y_pred = m.predict(X_test)
        pes = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        odob = sum(y_pred) / len(y_test)
        odobs.append(odob)
        scs.append(sc)
        pess.append(pes)
        recalls.append(recall * 6)
    x = np.arange(7)
    df = pd.DataFrame({'x': x, 'Accuracy': scs, 'Precision': pess, 'Racall': recalls, 'Одобренных': odobs})
    # df = df.sort_values(by=['Precision'])
    x = df['x'].to_numpy()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1.bar(x, df['Accuracy'], width=0.25, label='accuracy')
    # ax1.bar(x + 0.25, df['Precision'], width=0.25, label='precision')
    # ax2.bar(x + 0.50, df['Racall'], width=0.25, label='recall', color='g')
    ax1.bar(x, df['Accuracy'], width=0.25, label='Точность')
    ax1.bar(x + 0.25, df['Precision'], width=0.25, label='Точность Одобрений')
    ax2.bar(x + 0.50, df['Одобренных'], width=0.25, label='Доля одобренных', color='green')
    ax1.set_yticks(np.linspace(0.1, 1, 10))
    ax2.set_ylim(0, 0.1)
    # plt.bar(x + 0.45, df['Одобренных'], width=0.15, label='Одобренных')
    plt.xticks(x)
    plt.xlabel('Кейсы')
    plt.ylabel('Метрики')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # plt.tight_layout()
    plt.title('Кейсы и их эффективность')

    plt.savefig('Кейсы и их эффективность')


def test_model_decision_tree(p):
    target = 'ander'
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    X: pd.DataFrame = df.drop([target], 1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
                                                        random_state=1, stratify=None)

    # m = DecisionTreeClassifier(class_weight={0: 1, 1: 0.588},  # criterion='entropy',
    #                            max_depth=60, max_leaf_nodes=56, min_samples_leaf=7,
    #                            min_weight_fraction_leaf=0, random_state=5)

    m = DecisionTreeClassifier(max_depth=35, max_leaf_nodes=34, min_samples_leaf=9,
                               random_state=7)
    check_model_sklearn_cross(m, X_train, y_train)

    X_test, y_test = downsample(X_test, y_test, cl=1, rate=0.73)  # 0.28
    print("одоб test", sum(y_test))
    print("отклоненных test", len(y_test) - sum(y_test))
    check_model_sklearn_split(m, X_train, y_train, X_test, y_test)
    # -- decision tree
    import graphviz
    from sklearn.tree import export_graphviz, plot_tree
    dot_data = export_graphviz(m, out_file=None,
                               filled=True, rounded=True, special_characters=True,
                               feature_names=X.columns, class_names=['Отклонено', 'Одобрено'])
    graphviz.Source(dot_data).render('decision_tree')  # .save('aa.png')
    print('saved ' + 'decision_tree')
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,10))
    # plot_tree(results.best_estimator_, max_depth=4, filled=True, proportion=False, feature_names=X.columns)
    # plt.show()


def test_model_change_graph(p1, p2):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    target = 'ander'
    # -- data
    df_train: pd.DataFrame = pd.read_pickle(p1)
    df_test: pd.DataFrame = pd.read_pickle(p2)
    print(df_train.describe().to_string())
    print(df_test.describe().to_string())

    # df = remove_special_cases(df)
    # print(df.shape)
    # X: pd.DataFrame = df.drop([target, 'id'], 1)
    # X = StandardScaler().fit_transform(X)  # XGB specific
    # y = df[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False,
    #                                                     random_state=1, stratify=None)
    X_train: pd.DataFrame = df_train.drop(columns=[target, 'id'])
    X_test: pd.DataFrame = df_test.drop(columns=[target, 'id'])

    # print(X_train.columns.tolist())
    # print(X_train.tail(10).to_string())
    # exit()

    y_train = df_train[target]
    y_test = df_test[target]

    # X_train = StandardScaler().fit_transform(X_train)  # XGB specific
    # X_test = StandardScaler().fit_transform(X_test)  # XGB specific

    # df2: pd.DataFrame = pd.read_pickle('encoded_with_otcl.pickle')
    # df2 = df2[df.columns]
    # print(df2.shape)
    # df2: pd.DataFrame = pd.read_pickle(p)
    # X2: pd.DataFrame = df2.drop([target], 1)
    # X2 = StandardScaler().fit_transform(X2)  # XGB specific
    # y2 = df2[target]
    # X_train2, X_test, y_train2, y_test = train_test_split(X2, y2, test_size=0.20, shuffle=False,
    #                                                     random_state=1, stratify=None)

    x = []
    accur = []
    prec = []
    recall = []
    odobs = []

    for we in np.linspace(0.1, 3, 20).tolist():
        # est = DecisionTreeClassifier(class_weight={0: 1, 1: we}, # 0.588 # criterion='entropy',
        #                            max_depth=60, max_leaf_nodes=56, min_samples_leaf=7,
        #                            min_weight_fraction_leaf=0, random_state=5)
        # est = DecisionTreeClassifier(class_weight={0: 1, 1: we}, max_depth=10,
        #                              max_leaf_nodes=30, min_samples_leaf=6, random_state=7)
        # est = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                     colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
        #                     gamma=2, gpu_id=0, importance_type='gain',
        #                     interaction_constraints='', learning_rate=0.300000012,
        #                     max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
        #                     monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
        #                     num_parallel_tree=1, random_state=22, reg_alpha=0.2, reg_lambda=1,
        #                     scale_pos_weight=we, seed=22, subsample=1,
        #                     tree_method='exact', use_label_encoder=False,
        #                     validate_parameters=1, verbosity=1)
        # est = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                   colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0,
        #                   importance_type='gain', interaction_constraints='',
        #                   learning_rate=0.02, max_delta_step=0, max_depth=6,
        #                   min_child_weight=1, missing=np.nan, monotone_constraints='()',
        #                   n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
        #                   random_state=42, reg_alpha=0.5, reg_lambda=1,
        #                   scale_pos_weight=we, seed=42, subsample=1, # 1.3346062351320898
        #                   tree_method='exact', use_label_encoder=False,
        #                   validate_parameters=1, verbosity=1)
        # -- own --
        # est = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                     colsample_bynode=1, colsample_bytree=0.8, eval_metric='logloss',
        #                     gamma=2, gpu_id=0, importance_type='gain',
        #                     interaction_constraints='', learning_rate=0.300000012,
        #                     max_delta_step=0, max_depth=2, min_child_weight=6, missing=np.nan,
        #                     monotone_constraints='()', n_estimators=140, n_jobs=2, nthread=2,
        #                     num_parallel_tree=1, random_state=7, reg_alpha=0.2, reg_lambda=1,
        #                     scale_pos_weight=we, seed=7, subsample=1, tree_method='exact',
        #                     use_label_encoder=False, validate_parameters=1, verbosity=1)
        est = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',
                      gamma=2, gpu_id=0, importance_type='gain',
                      interaction_constraints='', learning_rate=0.300000012,
                      max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
                      monotone_constraints='()', n_estimators=270, n_jobs=2, nthread=2,
                      num_parallel_tree=1, random_state=7, reg_alpha=0.5, reg_lambda=1,
                      scale_pos_weight=we, seed=7, subsample=1, tree_method='exact',
                      use_label_encoder=False, validate_parameters=1, verbosity=1)

        # -- train
        est.fit(X_train, y_train)
        # -- score test
        sc = est.score(X_test, y_test)  # accuracy by test
        y_pred = est.predict(X_test)  #
        pes = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        odob = sum(y_pred) / len(y_test)

        print(we)
        print("Одобренных: %f" % odob)
        print("Accuracy: %f" % sc)
        print("Precision: %f" % pes)
        print("Recall: %f" % rec)
        print()
        x.append(we)
        accur.append(sc)
        prec.append(pes)
        recall.append(rec)
        odobs.append(odob)

    f, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, accur, label='Точность')
    ax.plot(x, prec, label='Точно одобренных')
    # ax.plot(x, recall, label='Recall')
    ax.plot(x, odobs, label='Доля одобрений')
    # ax.axvline(0.75)
    # ax.plot(0.3, 0.71, 'go', label='модель на ДеревьяхРешений')
    # ax.plot(1, 0.57, 'bo', label='модель на XGBoost')
    ax.plot(0.24, 0.69, 'bo', label='вероятностный классиф-ор')

    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    loc = plticker.MultipleLocator(base=0.05)
    ax.yaxis.set_major_locator(loc)

    plt.grid()
    plt.xlabel('Вес одобренных заявок')
    plt.ylabel('Метрики')
    plt.legend()
    # plt.show()
    plt.savefig('change_dtree.png')


def test_model(p1, p2, random_state=42):
    """ on hold out """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    target = 'ander'
    # -- data
    df_train = load(p1)
    df_test = load(p2)

    # -- fix types
    df_train = df_train.astype(float)
    df_test = df_test.astype(float)

    cols=[
        'EQUIFAX_RATING_SCORING_КИ отсутствует',
        'OKB_RATING_SCORING_Плохая КИ',
        'OKB_RATING_SCORING_Ошибка выполнения',
        'OKB_RATING_SCORING_Нейтральная КИ',
        '0prov_change_history',
        'МБКИ_кат_и_срок',
        'МБКИ_налсуд',
        'okbnh_MA_EMP',
        'МБКИ_налспецуч',
        'okbnh_MA_AS',
        'okbnh_MA_SWT',
        'okbnh_MA_PAS',
        'okbnh_MA_MS',
        'okbnh_MA_AS',
        'okbnh_MA_MS',
        '0prov_status_groupName_Действующее',
        'okbnh_MA_MTE',
        'okbnh_MA_REF',
        'NBKI_RATING_SCORING_Нейтральная КИ',
        'NBKI_RATING_SCORING_Ошибка выполнения',
        'okbnh_LCL_MA',
        # 'NBKI_RATING_SCORING_Хорошая КИ',
        'МБКИ_невыполнена',
        # 'CLIENT_AGE',
    ]

    # print(df_test.dtypes)
    # print(df_test.isna().sum())

    # df = remove_special_cases(df)

    print("Всего train:", df_train.shape)
    print("Отклонено", df_train[df_train[target] == 0].shape)
    print("Одобрено", df_train[df_train[target] == 1].shape)
    # return
    X_train: pd.DataFrame = df_train.drop(columns=[target])
    X_test: pd.DataFrame = df_test.drop(columns=[target])

    print(X_train.columns.tolist())

    s = [
        'ex4scor_scoring',
        'EQUIFAX_SCORING',
        'OKB_RATING_SCORING_КИ отсутствует',
        'CLIENT_WI_EXPERIENCE',
        'NBKI_SCORING',
        'afsnbki_scoring',
        'AUTO_DEAL_INITIAL_FEE',
        'OKB_RATING_SCORING_Хорошая КИ',
        'CLIENT_MARITAL_STATUS',
        'OKB_SCORING',
        'ANKETA_SCORING',
        'МБКИ_требаналотч',
        'МБКИ_нет_огр',
        '0prov_legalAddresses',
        'МБКИ_треб_исп_пр',
        '0prov_status_type',
        'МБКИ_треб_адрес',
        'CLIENT_AGE',
        'nbki_biom_resp_matchResults',
        'nbki_biom_resp_matchImages',
        'NBKI_RATING_SCORING_КИ отсутствует',
        'NBKI_RATING_SCORING_Хорошая КИ',
        # 'okbnh_MA_RAD',
        # 'okbnh_MA_SPA',
        # 'День недели',
        # 'okbnh_MA_SAM',
        # 'EQUIFAX_RATING_SCORING_КИ отсутствует',
        # 'CLIENT_GENDER',
        # '0prov_okfs_code',
        # '0prov_status_groupName_Действующее',
        #
        # 'okbnh_MA_SPE',
        # '0prov_status_isActing',
        # 'NBKI_RATING_SCORING_Нейтральная КИ',
        # 'okbnh_MULT_M',
        # 'EQUIFAX_RATING_SCORING_Хорошая КИ',
        # 'CLIENT_DEPENDENTS_COUNT',
        # 'okbnh_LCL_MA',
        # 'nbki_biom_resp_match_avg',
        # '0prov_includeInList',
        # '0prov_status_code',
        # 'NBKI_RATING_SCORING_Ошибка выполнения',
        # 'OKB_RATING_SCORING_Нейтральная КИ',
        # 'okbnh_MA_SMT',
        # 'okbnh_MA_SWT',
        # 'EQUIFAX_RATING_SCORING_Нейтральная КИ',
    ]

    y_train = df_train[target]
    y_test = df_test[target]

    print("after", X_train.columns.tolist())
    # print(X_train.isna().sum().to_string())

    # X_train = StandardScaler().fit_transform(X_train)  # XGB specific
    # X_test = StandardScaler().fit_transform(X_test)  # XGB specific


    # -- best model old
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=0, eval_metric='logloss',
    #                   importance_type='gain', interaction_constraints='',
    #                   learning_rate=0.02, max_delta_step=0, max_depth=6,
    #                   min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #                   n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
    #                   random_state=random_state, reg_alpha=0.5, reg_lambda=1,
    #                   seed=42, subsample=1, scale_pos_weight=0.28,  # 1.3346062351320898 # 0.36
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
    # -- all columns
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
    #               eval_metric='logloss', gamma=0.1, gpu_id=0, importance_type=None,
    #               interaction_constraints='', learning_rate=0.05, max_delta_step=0,
    #               max_depth=28, min_child_weight=1, missing=np.nan,
    #               monotone_constraints='()', n_estimators=122, n_jobs=2, nthread=4,
    #               num_parallel_tree=1, predictor='auto', random_state=42,
    #               reg_alpha=0.6, reg_lambda=1, scale_pos_weight=0.28, seed=42,
    #               subsample=1, tree_method='exact', use_label_encoder=False)
    # -- filtered columns
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
    #               eval_metric='logloss', gamma=0, gpu_id=0, importance_type=None,
    #               interaction_constraints='', learning_rate=0.03, max_delta_step=0,
    #               max_depth=10, min_child_weight=1, missing=np.nan,
    #               monotone_constraints='()', n_estimators=132, n_jobs=2, nthread=3,
    #               num_parallel_tree=1, predictor='auto', random_state=42,
    #               reg_alpha=0.6, reg_lambda=1, scale_pos_weight=0.28, seed=42,
    #               subsample=1, tree_method='exact', use_label_encoder=False)
    # -- own
    m = XGBClassifier(base_score=0.3, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                      eval_metric='logloss', gamma=0.1, gpu_id=0, importance_type=None,
                      interaction_constraints='', learning_rate=0.03, max_delta_step=0,
                      max_depth=20, min_child_weight=1, missing=np.nan,
                      monotone_constraints='()', n_estimators=152, n_jobs=2, nthread=4,
                      num_parallel_tree=1, predictor='auto', random_state=41,
                      reg_alpha=0.6, reg_lambda=1, scale_pos_weight=0.28, seed=41,
                      subsample=0.3, tree_method='exact', use_label_encoder=False)
    # m = RandomForestClassifier(class_weight={0: 1, 1: 1}, max_depth=6,
    #                            max_features='sqrt', n_estimators=80, random_state=7)
    # m = DecisionTreeClassifier(class_weight={0: 1, 1: 0.588},  # criterion='entropy',
    #                            max_depth=60, max_leaf_nodes=56, min_samples_leaf=7,
    #                            min_weight_fraction_leaf=0, random_state=5)
    # m = DecisionTreeClassifier(max_depth=35, max_leaf_nodes=34, min_samples_leaf=9,
    #                        random_state=7, class_weight={0: 1, 1: 1}) #)

    # X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy(), test_size=0.20, shuffle=False,
    #                                                     random_state=1, stratify=None)

    print("train одоб ", sum(y_train))
    print("train отклоненных", len(y_train) - sum(y_train))
    print("train 1/0", sum(y_train) / (len(y_train) - sum(y_train)))
    train_rate = sum(y_train) / (len(y_train) - sum(y_train))
    print("test одоб", sum(y_test))
    print("test отклоненных", len(y_test) - sum(y_test))
    print("test 1/0", sum(y_test) / (len(y_test) - sum(y_test)))

    print()
    m.fit(X_train, y_train)

    check_model_sklearn_cross(m, X_train, y_train, splits=5, calc_confusion_matrix=False)

    X_test, y_test = downsample(X_test, y_test, rate=train_rate)
    print("test shapes", X_test.shape, y_test.shape)
    print("train shapes", X_train.shape, y_train.shape)
    print("1/0 after", sum(y_test) / (len(y_test) - sum(y_test)))
    print("test одоб", sum(y_test))
    print("test отклоненных", len(y_test) - sum(y_test))
    print("test 1/0", sum(y_test) / (len(y_test) - sum(y_test)))
    check_model_sklearn_split(m, X_train, y_train, X_test, y_test, calc_confusion_matrix=False)


def test_model_regression(p):
    """ on hold out """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    target = 'ander'
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    print("Всего:", df.shape)
    print("Отклонено", df[df[target] == 0].shape)
    print("Одобрено", df[df[target] == 1].shape)
    p = 'after_read.pickle'
    df_origin: pd.DataFrame = pd.read_pickle(p)
    ind_relese = df_origin[df_origin['DEAL_STATUS'] == 'release'].index
    inter = np.intersect1d(ind_relese, df.index)
    df.loc[inter, 'ander'] = 1.5
    # print(df.loc[ind_relese])
    # print(df['ander'].unique())
    # return

    X: pd.DataFrame = df.drop([target], 1)

    y = df[target]
    # X = StandardScaler().fit_transform(X)  # XGB specific
    m = DecisionTreeRegressor(max_depth=35, max_leaf_nodes=34, min_samples_leaf=9,
                              random_state=7)  # class_weight={0: 1, 1: 1}
    X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy(), test_size=0.20, shuffle=False,
                                                        random_state=1, stratify=None)

    print("одоб test", sum(y_test))
    print("отклоненных test", len(y_test) - sum(y_test))
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import cross_validate, cross_val_score
    splits = 5
    kfold = StratifiedKFold(n_splits=splits)
    m.fit(X_train, y_train)
    results = cross_val_score(m, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    print("mean_squared_error", results.mean())

    # -- decision tree
    import graphviz
    from sklearn.tree import export_graphviz, plot_tree
    dot_data = export_graphviz(m, out_file=None,
                               filled=True, rounded=True, special_characters=True,
                               feature_names=X.columns, class_names=['Отклонено', 'Одобрено'])
    graphviz.Source(dot_data).render('decision_tree')  # .save('aa.png')
    print('saved ' + 'decision_tree')

    # print("Accuracy: %f" % results['test_score'].mean())
    # print(results)

    # check_model_sklearn_cross(m, X_train, y_train)
    #
    # X_test, y_test = downsample(X_test, y_test, cl=1, rate=0.73)  # 0.28
    # print("одоб test", sum(y_test))
    # print("отклоненных test", len(y_test) - sum(y_test))
    # check_model_sklearn_split(m, X_train, y_train, X_test, y_test)


def replace_names(p, fields_dict):
    df: pd.DataFrame = pd.read_pickle(p)
    df = rename_columns(df, columns=fields_dict, len=99)  # for Evseeva
    return save('replace_names.pickle', df)


def pca_clustering_3(p, i=1, n_clusters=2):
    """
    required prepare_for_clustering or standartization

    На PCA или не на PCA легко переключить, отправив алгоритму PCA principals или df

    :param p:
    :return:
    """
    from sklearn.decomposition import KernelPCA
    from sklearn import metrics
    df_o: pd.DataFrame = pd.read_pickle(p)
    df_o = df_o.drop(columns=['id'])

    df_o.reset_index(inplace=True, drop=True)
    # weight for feature
    ander = df_o['ander'].to_numpy().copy()
    # df_o = df_o[df_o['ander'] == 1]
    # print(df_o['ander'].unique())
    df_o['ander'] = ander * i
    # df_o['EQUIFAX_RATING_SCORING_Хорошая КИ'] = df_o['EQUIFAX_RATING_SCORING_Хорошая КИ'] *1.2
    # df_o['EQUIFAX_RATING_SCORING_КИ отсутствует'] = df_o['EQUIFAX_RATING_SCORING_КИ отсутствует'] * 1.2
    # df_o['CLIENT_AGE'] = df_o['CLIENT_AGE'] * 0.0001
    # df['ANKETA_SCORING'] = df['ANKETA_SCORING'] * 0.0001
    df_o['AUTO_DEAL_INITIAL_FEE'] = df_o['AUTO_DEAL_INITIAL_FEE'] * 0.00001
    # df['EQUIFAX_SCORING'] = df['EQUIFAX_SCORING'] * 0.00001
    df_o['NBKI_SCORING'] = df_o['NBKI_SCORING'] * 1.2
    df = df_o.copy()

    # -- PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    pca = KernelPCA(
        # kernel='rbf',
        kernel='linear',
        # kernel='sigmoid',
        # kernel='cosine',
        n_components=2)

    X_principal = pca.fit_transform(df)
    # -- clusters on PCA
    # linkage = 'single'
    linkage = 'average'
    # linkage='complete'
    # linkage = 'ward'
    #
    # affinity = "manhattan"
    # affinity='manhattan'
    affinity = 'euclidean'

    # ac = AgglomerativeClustering(
    #     linkage=linkage,
    #     # affinity=affinity,
    #     n_clusters=n_clusters,
    #     compute_full_tree=False,
    #     compute_distances=True)
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import OPTICS
    from sklearn.cluster import Birch
    # ac = AffinityPropagation(damping=0.5) #, random_state=0, max_iter=50, verbose=True)
    # ac = OPTICS()
    # ac = Birch(n_clusters=3)
    # from sklearn.cluster import KMeans
    # ac = KMeans(n_clusters=4)
    from sklearn.mixture import GaussianMixture
    ac = GaussianMixture(n_components=n_clusters, random_state=3)
    # ac = AgglomerativeClustering(n_clusters=3)

    # -- cluster
    labels = ac.fit_predict(X_principal)
    # labels = ac.fit_predict(df)

    # -- clusters clusters
    from sklearn.preprocessing import scale, minmax_scale
    # labels = scale(labels)

    # labels = minmax_scale(labels,(-1,1))
    # weight = 1
    nc = 'cluster'
    # df[nc] = labels #* weight
    # df = pd.get_dummies(df, columns=[nc], dummy_na=False)
    #
    # cl_cols = [c for c in df.columns if c.startswith(nc)]
    # for c in cl_cols:
    #     df[c] = scale(df[c])

    # -- cluster clusters
    linkage = 'single'
    # linkage = 'average'
    # linkage = 'complete'
    # linkage = 'ward'
    # affinity = "manhattan"
    # affinity = 'cosine'
    # affinity = 'euclidean'

    # ac = AgglomerativeClustering(n_clusters=6, linkage=linkage, affinity=affinity)
    # ac = GaussianMixture(n_components=4, random_state=3)
    # labels = ac.fit_predict(X_principal2)
    df[nc] = labels
    # df[df[nc] == 8] = 1
    # df[df[nc] == 2] = 1
    #
    # df[df[nc] == 3] = 4
    # df[df[nc] == 5] = 4

    df[df[nc] == 2] = 1

    print(df[nc].tolist())

    labels = df[nc].tolist()
    # labels = [4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 1, 4, 0, 4, 1, 0, 1, 1, 1, 4, 4, 0, 4, 1, 4, 0, 1, 4, 0, 4, 4, 0, 4, 4, 0, 1, 4, 1, 0, 4, 1, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 1, 0, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 1, 4, 4, 0, 1, 4, 4, 4, 4, 4, 1, 4, 0, 4, 1, 0, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 1, 0, 4, 4, 4, 0, 4, 4, 4, 4, 1, 4, 0, 1, 4, 1, 4, 4, 0, 4, 0, 1, 0, 4, 1, 4, 0, 1, 4, 1, 4, 4, 1, 4, 4, 1, 4, 0, 4, 4, 0, 4, 4, 1, 1, 0, 1, 4, 1, 4, 0, 4, 0, 4, 0, 0, 4, 4, 4, 1, 4, 4, 1, 4, 4, 4, 1, 0, 0, 1, 4, 4, 4, 4, 1, 0, 4, 0, 0, 4, 1, 0, 4, 1, 1, 1, 4, 0, 1, 1, 0, 0, 0, 0, 4, 1, 1, 4, 1, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 1, 4, 0, 1, 4, 4, 4, 0, 1, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4, 4, 4, 4, 1, 4, 4, 4, 4, 0, 1, 1, 0, 0, 4, 0, 4, 1, 1, 4, 0, 4, 1, 1, 4, 4, 4, 4, 1, 4, 4, 1, 0, 1, 1, 4, 4, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 1, 1, 0, 4, 4, 1, 0, 4, 4, 1, 4, 4, 4, 1, 4, 4, 1, 1, 1, 0, 4, 1, 4, 1, 0, 0, 4, 0, 0, 4, 1, 1, 4, 1, 4, 1, 4, 0, 4, 0, 4, 1, 4, 0, 4, 1, 1, 4, 0, 4, 1, 1, 0, 4, 0, 1, 1, 4, 0, 1, 1, 1, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 1, 1, 4, 4, 1, 1, 4, 4, 4, 0, 1, 4, 4, 4, 0, 1, 1, 4, 4, 4, 0, 1, 1, 4, 4, 4, 4, 4, 4, 0, 0, 1, 4, 4, 0, 4, 1, 4, 1, 1, 0, 0, 1, 1, 4, 4, 0, 0, 0, 1, 4, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 1, 0, 1, 1, 0, 4, 4, 1, 4, 4, 0, 4, 1, 4, 1, 4, 4, 4, 4, 1, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 4, 1, 4, 4, 4, 0, 1, 1, 4, 1, 4, 1, 0, 4, 4, 0, 0, 1, 4, 0, 4, 1, 0, 4, 4, 1, 0, 4, 4, 1, 4, 4, 0, 4, 4, 0, 4, 0, 4, 4, 4, 0, 0, 0, 4, 4, 0, 4, 4, 4, 1, 4, 1, 4, 0, 1, 4, 1, 0, 4, 0, 4, 4, 1, 4, 4, 4, 1, 4, 1, 0, 4, 0, 4, 4, 0, 0, 1, 1, 4, 0, 4, 4, 1, 0, 4, 4, 0, 4, 4, 4, 1, 0, 4, 1, 4, 0, 4, 4, 4, 0, 4, 0, 0, 4, 1, 4, 4, 0, 1, 4, 0, 1, 4, 4, 1, 4, 4, 1, 4, 1, 1, 4, 4, 1, 1, 0, 1, 4, 0, 0, 1, 4, 0, 0, 1, 4, 4, 0, 1, 4, 0, 4, 4, 4, 0, 1, 4, 1, 0, 4, 0, 4, 4, 1, 4, 4, 1, 1, 1, 4, 0, 1, 0, 4, 4, 1, 4, 4, 0, 0, 4, 0, 4, 4, 1, 0, 4, 0, 4, 0, 4, 1, 1, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 4, 0, 4, 4, 0, 4, 4, 0, 1, 0, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 0, 4, 4, 1, 1, 1, 4, 0, 0, 1, 4, 1, 4, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 0, 4, 0, 0, 0, 4, 4, 4, 0, 4, 4, 0, 1, 1, 4, 4, 1, 4, 0, 1, 4, 0, 4, 4, 1, 4, 0, 4, 1, 4, 4, 0, 1, 4, 1, 4, 4, 1, 1, 4, 4, 4, 4, 4, 0, 1, 0, 4, 4, 4, 0, 1, 4, 4, 1, 0, 4, 0, 0, 4, 4, 1, 1, 4, 1, 4, 4, 0, 4, 4, 4, 0, 1, 0, 0, 1, 1, 1, 1, 0, 4, 1, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 0, 1, 4, 4, 4, 4, 1, 4, 4, 0, 4, 1, 4, 4, 0, 4, 1, 4, 0, 0, 4, 0, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 4, 0, 4, 1, 4, 4, 4, 1, 0, 4, 4, 0, 1, 4, 0, 1, 0, 1, 1, 4, 4, 1, 4, 1, 4, 0, 4, 1, 4, 4, 4, 1, 4, 4, 0, 1, 0, 0, 4, 4, 4, 1, 1, 1, 1, 4, 4, 0, 4, 4, 1, 4, 4, 4, 4, 0, 4, 4, 1, 4, 4, 4, 1, 0, 4, 4, 0, 0, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 0, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 0, 1, 4, 1, 1, 0, 0, 4, 0, 0, 0, 0, 4, 4, 1, 0, 4, 4, 4, 4, 4, 0, 1, 0, 0, 0, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 0, 0, 1, 0, 4, 0, 0, 4, 4, 0, 0, 1, 4, 4, 0, 1, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 0, 0, 4, 1, 4, 0, 4, 4, 1, 1, 4, 1, 1, 0, 0, 4, 4, 1, 1, 4, 4, 0, 4, 0, 4, 4, 1, 1, 4, 4, 1, 0, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 1, 1, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 1, 4, 0, 0, 4, 4, 4, 0, 0, 4, 4, 0, 4, 1, 0, 1, 4, 0, 0, 4, 4, 4, 1, 4, 1, 4, 4, 1, 0, 4, 4, 4, 4, 4, 4, 1, 1, 4, 0, 0, 4, 1, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1, 4, 0, 4, 0, 4, 4, 4, 1, 1, 1, 1, 0, 4, 0, 1, 1, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 1, 1, 4, 4, 1, 4, 4, 0, 4, 1, 1, 4, 4, 0, 1, 4, 0, 1, 4, 4, 4, 0, 0, 1, 0, 1, 1, 0, 4, 0, 4, 4, 1, 0, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 1, 4, 1, 1, 4, 0, 1, 0, 1, 4, 4, 0, 0, 4, 0, 1, 1, 1, 4, 4, 0, 1, 1, 4, 1, 4, 4, 4, 1, 1, 0, 4, 0, 0, 4, 0, 1, 4, 4, 0, 4, 1, 0, 1, 0, 0, 4, 1, 1, 4, 1, 4, 1, 1, 1, 1, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 0, 1, 0, 0, 4, 0, 4, 1, 0, 4, 1, 4, 4, 0, 1, 4, 1, 4, 0, 0, 4, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 1, 1, 0, 4, 0, 1, 4, 4, 0, 0, 0, 4, 4, 0, 4, 4, 0, 1, 1, 4, 1, 4, 1, 4, 0, 1, 4, 1, 4, 1, 0, 4, 0, 4, 0, 0, 4, 0, 4, 1, 4, 0, 0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 0, 1, 0, 4, 4, 0, 0, 4, 1, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 0, 1, 4, 4, 4, 4, 4, 1, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 0, 1, 4, 4, 0, 0, 1, 4, 4, 4, 1, 0, 0, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 4, 4, 4, 1, 4, 1, 4, 4, 1, 0, 4, 4, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 0, 4, 4, 1, 1, 4, 4, 4, 1, 1, 1, 0, 0, 0, 4, 0, 1, 4, 0, 0, 4, 0, 0, 4, 1, 4, 1, 4, 1, 1, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 0, 1, 1, 0, 4, 1, 4, 0, 4, 4, 4, 4, 0, 4, 1, 4, 0, 1, 4, 4, 1, 4, 4, 1, 0, 0, 0, 0, 1, 4, 4, 4, 1, 1, 4, 1, 0, 4, 1, 4, 4, 0, 0, 4, 1, 1, 0, 0, 1, 1, 4, 4, 4, 1, 0, 4, 4, 1, 4, 4, 0, 1, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 4, 4, 0, 0, 4, 1, 4, 0, 0, 0, 4, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 0, 4, 0, 4, 4, 0, 1, 4, 0, 1, 4, 1, 1, 4, 0, 0, 0, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 1, 0, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 1, 4, 4, 0, 4, 0, 0, 1, 4, 1, 1, 1, 4, 0, 4, 4, 4, 4, 4, 0, 1, 0, 1, 4, 4, 4, 4, 0, 4, 0, 4, 1, 1, 4, 1, 4, 4, 1, 0, 4, 4, 1, 4, 4, 0, 4, 1, 1, 4, 4, 0, 1, 1, 4, 4, 4, 4, 0, 1, 4, 1, 1, 1, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 1, 4, 4, 4, 0, 0, 0, 1, 4, 4, 4, 0, 0, 0, 4, 0, 1, 1, 4, 4, 1, 0, 1, 4, 1, 1, 4, 0, 4, 0, 4, 4, 0, 0, 0, 0, 1, 1, 4, 1, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 0, 4, 1, 4, 1, 4, 1, 4, 4, 4, 0, 4, 4, 1, 4, 0, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 0, 4, 4, 1, 0, 1, 4, 4, 0, 0, 0, 1, 4, 4, 0, 1, 4, 0, 0, 4, 1, 1, 0, 4, 4, 4, 1, 4, 0, 4, 0, 0, 4, 4, 4, 1, 0, 4, 4, 0, 1, 1, 0, 4, 1, 4, 4, 1, 4, 4, 0, 4, 0, 4, 0, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 0, 1, 0, 4, 0, 1, 4, 4, 4, 1, 0, 0, 4, 1, 1, 4, 4, 4, 4, 0, 1, 4, 0, 0, 4, 4, 4, 1, 4, 1, 4, 0, 0, 4, 1, 4, 1, 1, 1, 0, 1, 4, 4, 4, 0, 4, 0, 0, 4, 4, 0, 4, 4, 1, 0, 0, 4, 1, 4, 4, 4, 0, 1, 1, 4, 0, 4, 0, 0, 4, 1, 4, 1, 1, 0, 0, 0, 4, 4, 1, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 0, 0, 4, 1, 4, 4, 1, 4, 4, 4, 1, 1, 4, 4, 0, 4, 0, 1, 1, 4, 4, 0, 4, 0, 4, 1, 1, 4, 1, 4, 1, 4, 1, 0, 4, 0, 1, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 0, 4, 1, 4, 0, 4, 1, 1, 0, 1, 0, 4, 0, 4, 1, 0, 1, 4, 4, 0, 0, 4, 0, 1, 4, 1, 4, 0, 1, 1, 1, 4, 1, 4, 4, 0, 4, 0, 4, 4, 4, 4, 1, 4, 4, 1, 4, 1, 1, 4, 4, 4, 0, 4, 4, 1, 1, 0, 4, 4, 0, 1, 1, 1, 1, 0, 0, 4, 4, 0, 4, 4, 4, 1, 4, 4, 0, 0, 1, 0, 4, 0, 0, 0, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 0, 4, 4, 0, 1, 4, 4, 4, 1, 1, 0, 4, 4, 4, 1, 4, 1, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 1, 0, 4, 0, 0, 4, 1, 1, 0, 4, 1, 0, 0, 4, 0, 4, 0, 1, 0, 4, 4, 0, 4, 4, 1, 1, 1, 4, 1, 0, 0, 4, 0, 4, 1, 1, 4, 1, 4, 0, 0, 0, 1, 0, 1, 0, 4, 0, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4, 4, 1, 1, 1, 4, 4, 1, 4, 0, 0, 4, 4, 0, 0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 1, 0, 0, 0, 1, 1, 1, 0, 4, 1, 1, 4, 1, 1, 1, 0, 1, 1, 4, 4, 0, 0, 4, 0, 0, 0, 4, 1, 1, 1, 4, 0, 1, 4, 4, 0, 1, 4, 4, 0, 4, 1, 4, 0, 4, 0, 4, 4, 4, 0, 1, 4, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 0, 0, 4, 4, 1, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 1, 0, 4, 4, 0, 0, 1, 4, 4, 0, 4, 4, 1, 4, 4, 1, 4, 0, 4, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 4, 1, 4, 4, 0, 4, 1, 4, 0, 1, 4, 4, 4, 4, 1, 0, 4, 1, 4, 4, 0, 1, 0, 0, 1, 0, 1, 4, 4, 4, 4, 0, 1, 0, 4, 0, 4, 1, 4, 0, 0, 0, 1, 4, 0, 4, 1, 4, 4, 1, 0, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4, 0, 4, 4, 0, 1, 4, 4, 1, 4, 1, 4, 4, 1, 4, 4, 1, 4, 0, 4, 4, 1, 4, 4, 4, 4, 0, 4, 4, 0, 1, 4, 4, 4, 4, 1, 4, 4, 1, 0, 4, 4, 4, 0, 0, 0, 4, 4, 0, 1, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 0, 0, 4, 1, 4, 4, 0, 1, 4, 4, 1, 0, 1, 4, 4, 1, 0, 4, 0, 4, 1, 1, 4, 1, 1, 0, 4, 0, 0, 0, 0, 4, 4, 4, 0, 4, 1, 0, 4, 4, 0, 4, 4, 1, 4, 4, 0, 4, 4, 1, 4, 4, 0, 4, 4, 4, 1, 0, 4, 4, 1, 4, 4, 4, 1, 1, 4, 4, 0, 0, 4, 0, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 1, 1, 1, 0, 4, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 1, 0, 1, 4, 4, 4, 0, 1, 4, 4, 1, 4, 4, 4, 1, 1, 1, 4, 1, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 0, 4, 4, 0, 0, 4, 4, 4, 0, 1, 4, 1, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 0, 0, 0, 1, 0, 4, 1, 0, 4, 1, 4, 0, 4, 4, 0, 0, 1, 4, 4, 0, 4, 1, 0, 4, 4, 4, 4, 4, 0, 4, 0, 0, 4, 4, 4, 4, 1, 1, 1, 4, 1, 4, 0, 0, 4, 4, 1, 1, 4, 0, 4, 4, 4, 1, 1, 0, 1, 0, 4, 0, 1, 4, 4, 4, 4, 0, 4, 4, 4, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 0, 4, 0, 4, 1, 4, 4, 1, 1, 4, 1, 4, 4, 4, 1, 4, 4, 0, 4, 4, 4, 1, 1, 4, 4, 0, 1, 0, 0, 4, 4, 1, 4, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 1, 0, 1, 4, 0, 4, 0, 4, 1, 1, 4, 0, 4, 1, 0, 4, 0, 1, 4, 0, 4, 0, 0, 4, 4, 4, 0, 4, 4, 0, 1, 4, 4, 4, 1, 0, 4, 4, 4, 0, 0, 1, 1, 0, 1, 4, 0, 4, 4, 4, 0, 4, 4, 1, 1, 4, 4, 1, 4, 4, 0, 1, 1, 0, 1, 4, 1, 4, 1, 4, 0, 4, 4, 4, 4, 1, 4, 1, 1, 0, 4, 1, 4, 4, 1, 4, 4, 0, 4, 4, 0, 0, 1, 4, 4, 4, 1, 1, 0, 1, 1, 1, 4, 0, 4, 1, 0, 1, 4, 1, 0, 4, 4, 1, 0, 1, 0, 4, 4, 0, 0, 0, 0, 0, 4, 0, 4, 1, 0, 4, 4, 0, 4, 1, 1, 1, 4, 1, 0, 4, 1, 0, 1, 0, 4, 4, 1, 4, 0, 1, 4, 4, 1, 0, 4, 1, 1, 1, 1, 4, 0, 0, 1, 0, 4, 0, 4, 0, 0, 0, 4, 1, 1, 4, 4, 4, 4, 0, 0, 1, 0, 0, 4, 1, 4, 4, 0, 4, 1, 4, 4, 0, 0, 1, 0, 1, 0, 4, 0, 4, 4, 4, 4, 0, 0, 4, 4, 4, 1, 4, 0, 4, 0, 0, 0, 4, 4, 1, 4, 1, 1, 1, 4, 0, 4, 0, 1, 4, 0, 1, 0, 4, 0, 4, 1, 1, 1, 0, 1, 0, 4, 0, 0, 1, 4, 0, 0, 4, 4, 0, 4, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1, 0, 4, 0, 1, 4, 0, 1, 1, 4, 0, 4, 4, 4, 4, 1, 0, 4, 4, 0, 4, 4, 0, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 0, 4, 1, 4, 1, 1, 4, 4, 1, 4, 1, 4, 4, 0, 0, 4, 4, 1, 1, 4, 1, 4, 0, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 1, 4, 4, 1, 0, 4, 1, 0, 0, 0, 1, 4, 0, 4, 1, 4, 4, 4, 1, 0, 4, 0, 1, 4, 4, 1, 1, 0, 0, 0, 1, 4, 4, 4, 4, 0, 0, 1, 0, 1, 0, 4, 0, 4, 4, 4, 1, 0, 0, 4, 4, 1, 4, 4, 4, 0, 4, 1, 4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 1, 0, 1, 1, 0, 1, 1, 4, 4, 0, 1, 1, 4, 0, 1, 0, 4, 1, 4, 1, 1, 1, 0, 4, 4, 4, 4, 0, 0, 0, 4, 1, 4, 4, 4, 4, 1, 1, 4, 1, 0, 1, 4, 4, 1, 4, 4, 1, 1, 0, 4, 0, 0, 0, 0, 4, 0, 4, 1, 4, 4, 0, 4, 4, 4, 1, 1, 4, 0, 0, 4, 1, 0, 0, 0, 1, 1, 0, 4, 0, 4, 0, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 1, 1, 1, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 0, 4, 4, 1, 4, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 1, 1, 4, 0, 0, 1, 4, 4, 0, 4, 4, 4, 4, 1, 0, 1, 1, 4, 1, 0, 1, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 0, 1, 4, 4, 0, 1, 4, 4, 0, 4, 0, 4, 1, 1, 0, 0, 4, 4, 4, 0, 0, 1, 4, 1, 4, 0, 0, 4, 1, 0, 4, 4, 1, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0, 4, 4, 0, 4, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4, 4, 1, 1, 0, 4, 0, 0, 4, 0, 4, 1, 0, 4, 1, 1, 4, 1, 4, 4, 4, 0, 0, 1, 1, 0, 0, 1, 4, 0, 0, 1, 4, 4, 1, 4, 1, 1, 1, 1, 0, 4, 0, 1, 4, 4, 4, 0, 1, 1, 1, 4, 4, 0, 4, 1, 0, 4, 1, 1, 1, 1, 4, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 1, 0, 1, 1, 4, 4, 1, 0, 1, 4, 4, 4, 0, 1, 4, 4, 4, 1, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 1, 4, 4, 1, 4, 1, 4, 1, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 1, 0, 4, 1, 4, 4, 4, 4, 0, 1, 0, 1, 1, 1, 4, 4, 4, 4, 0, 4, 0, 1, 1, 4, 0, 4, 4, 1, 0, 4, 0, 4, 0, 4, 0, 1, 1, 1, 4, 0, 0, 4, 4, 1, 0, 4, 4, 4, 0, 1, 4, 4, 4, 0, 4, 4, 0, 4, 1, 4, 1, 0, 4, 1, 4, 4, 0, 0, 4, 1, 4, 1, 1, 4, 1, 4, 0, 1, 4, 4, 4, 0, 0, 4, 4, 0, 0, 4, 1, 4, 0, 4, 1, 4, 1, 1, 0, 4, 1, 1, 0, 1, 4, 4, 1, 4, 0, 4, 1, 1, 4, 0, 4, 0, 0, 1, 1, 0, 4, 0, 1, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 0, 0, 4, 4, 1, 0, 1, 1, 1, 4, 4, 4, 4, 4, 0, 0, 1, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 1, 0, 0, 1, 0, 0, 1, 4, 1, 4, 0, 4, 1, 0, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 0, 4, 1, 4, 0, 4, 1, 1, 4, 0, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 1, 1, 0, 4, 4, 1, 0, 0, 0, 4, 4, 4, 0, 4, 0, 1, 4, 0, 4, 4, 4, 4, 4, 0, 1, 4, 4, 1, 4, 1, 4, 4, 0, 0, 4, 4, 4, 1, 0, 1, 4, 4, 0, 1, 4, 4, 4, 4, 0, 0, 4, 1, 1, 4, 4, 4, 1, 4, 0, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4, 1, 1, 4, 1, 4, 4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0, 0, 0, 0, 1, 4, 4, 4, 0, 4, 0, 1, 4, 1, 4, 4, 4, 0, 4, 1, 4, 1, 4, 0, 0, 4, 1, 1, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 4, 4, 1, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 1, 4, 4, 0, 4, 4, 0, 4, 1, 0, 4, 4, 4, 1, 1, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 1, 1, 0, 1, 1, 4, 1, 4, 0, 1, 0, 0, 4, 4, 4, 1, 0, 0, 0, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 1, 0, 4, 1, 1, 4, 0, 1, 4, 1, 1, 0, 0, 4, 1, 4, 0, 0, 0, 1, 4, 0, 4, 4, 4, 1, 0, 4, 1, 4, 0, 4, 0, 4, 4, 4, 4, 1, 0, 0, 1, 4, 1, 1, 4, 4, 4, 1, 0, 1, 4, 0, 1, 4, 4, 4, 0, 4, 1, 1, 4, 4, 4, 0, 4, 4, 1, 4, 4, 1, 4, 4, 0, 0, 4, 4, 1, 0, 1, 1, 4, 1, 1, 4, 0, 1, 4, 4, 4, 0, 0, 0, 1, 1, 0, 0, 4, 0, 4, 1, 4, 4, 4, 4, 0, 1, 1, 1, 4, 0, 1, 1, 4, 4, 1, 4, 0, 0, 4, 1, 4, 1, 4, 0, 4, 4, 4, 0, 0, 1, 1, 1, 4, 1, 4, 4, 4, 4, 4, 1, 4, 1, 4, 4, 4, 1, 0, 0, 4, 4, 1, 4, 4, 4, 1, 0, 4, 1, 4, 0, 4, 1, 0, 0, 4, 4, 0, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 0, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 1, 0, 4, 4, 1, 4, 0, 4, 4, 0, 4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 1, 4, 4, 1, 1, 1, 0, 0, 4, 0, 0, 1, 0, 4, 1, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 1, 4, 4, 0, 4, 1, 4, 4, 0, 4, 4, 0, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 4, 0, 1, 4, 4, 1, 1, 0, 1, 4, 4, 4, 1, 0, 4, 4, 1, 4, 4, 0, 4, 1, 4, 4, 4, 4, 1, 4, 1, 1, 4, 4, 4, 0, 0, 0, 1, 0, 4, 1, 1, 4, 0, 1, 4, 0, 0, 4, 0, 0, 0, 0, 1, 4, 1, 1, 4, 4, 4, 4, 0, 4, 4, 1, 4, 1, 1, 0, 0, 0, 4, 1, 1, 4, 4, 4, 1, 0, 4, 4, 4, 1, 1, 0, 1, 4, 4, 0, 4, 4, 0, 1, 4, 1, 4, 1, 0, 0, 4, 0, 0, 0, 4, 0, 4, 1, 4, 1, 0, 0, 4, 0, 4, 0, 4, 4, 4, 4, 1, 4, 1, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 1, 4, 0, 0, 1, 4, 1, 0, 4, 1, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 1, 4, 4, 1, 0, 0, 4, 0, 4, 0, 0, 1, 1, 1, 0, 4, 4, 0, 4, 1, 4, 0, 4, 0, 1, 4, 0, 4, 0, 4, 4, 1, 4, 0, 4, 1, 4, 1, 1, 0, 4, 1, 4, 4, 0, 1, 4, 4, 4, 1, 1, 1, 0, 4, 0, 4, 1, 4, 4, 4, 0, 4, 4, 0, 1, 0, 4, 4, 4, 0, 0, 4, 4, 0, 1, 4, 4, 4, 4, 0, 4, 1, 0, 0, 1, 4, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 1, 1, 4, 1, 1, 0, 0, 1, 0, 0, 4, 4, 1, 0, 1, 0, 4, 4, 1, 1, 4, 1, 0, 4, 1, 4, 1, 4, 4, 4, 1, 0, 4, 4, 1, 1, 0, 4, 1, 1, 1, 1, 4, 4, 1, 1, 0, 4, 1, 0, 1, 4, 4, 1, 4, 4, 0, 1, 4, 4, 0, 4, 1, 4, 0, 0, 4, 4, 4, 4, 4, 1, 4, 4, 0, 4, 4, 1, 1, 1, 0, 4, 0, 0, 4, 4, 0, 4, 1, 0, 0, 4, 4, 1, 1, 4, 4, 4, 1, 0, 4, 0, 4, 1, 4, 4, 4, 1, 1, 0, 4, 1, 1, 4, 1, 4, 4, 4, 1, 4, 1, 4, 1, 0, 1, 4, 4, 4, 1, 0, 1, 4, 0, 4, 0, 0, 4, 0, 4, 4, 4, 4, 1, 0, 1, 4, 1, 1, 1, 0, 4, 0, 1, 4, 1, 0, 4, 4, 4, 4, 0, 1, 4, 1, 4, 1, 1, 1, 0, 1, 4, 1, 1, 0, 1, 4, 1, 4, 1, 4, 4, 4, 4, 0, 1, 0, 4, 1, 0, 1, 4, 1, 1, 4, 1, 0, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 1, 0, 1, 4, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 4, 0, 4, 4, 4, 0, 0, 1, 4, 0, 0, 1, 1, 1, 0, 1, 4, 4, 4, 1, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 4, 0, 4, 0, 4, 0, 4, 1, 1, 1, 4, 0, 4, 0, 1, 4, 4, 4, 1, 4, 1, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 4, 4, 1, 1, 0, 4, 4, 1, 4, 0, 4, 0, 4, 4, 1, 0, 1, 1, 4, 1, 4, 1, 4, 4, 1, 0, 0, 4, 4, 4, 1, 4, 0, 0, 0, 4, 0, 0, 4, 0, 1, 4, 1, 4, 4, 1, 4, 4, 4, 4, 1, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 1, 4, 4, 1, 4, 1, 1, 4, 4, 4, 4, 0, 0, 1, 4, 4, 0, 1, 4, 4, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 4, 0, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 1, 1, 0, 4, 4, 4, 1, 0, 1, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 0, 0, 4, 0, 0, 1, 1, 4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 1, 4, 0, 1, 4, 0, 1, 4, 4, 4, 4, 1, 4, 4, 1, 4, 0, 0, 4, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 0, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 1, 0, 4, 4, 4, 1, 1, 4, 4, 4, 1, 0, 4, 4, 0, 4, 1, 4, 1, 4, 0, 1, 0, 4, 4, 4, 0, 0, 1, 4, 4, 4, 4, 4, 4, 1, 0, 0, 4, 0, 0, 1, 4, 1, 4, 0, 4, 4, 0, 1, 4, 4, 4, 4, 1, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 4, 0, 4, 4, 1, 4, 4, 4, 0, 1, 1, 0, 4, 4, 1, 4, 4, 1, 4, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 0, 4, 1, 1, 4, 4, 4, 4, 1, 1, 1, 0, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 0, 4, 1, 1, 1, 0, 4, 4, 4, 0, 4, 0, 1, 1, 4, 0, 4, 0, 1, 0, 4, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 1, 4, 1, 4, 4, 1, 0, 1, 4, 1, 1, 4, 0, 1, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 1, 4, 4, 4, 4, 0, 1, 4, 4, 4, 4, 4, 0, 4, 1, 1, 4, 0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 4, 4, 1, 4, 0, 0, 4, 1, 4, 4, 0, 4, 1, 4, 4, 1, 1, 1, 4, 4, 0, 4, 0, 4, 4, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1, 1, 1, 4, 0, 0, 4, 4, 1, 4, 0, 1, 4, 1, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 4, 1, 0, 4, 4, 1, 4, 1, 0, 0, 0, 0, 4, 4, 4, 1, 4, 4, 4, 4, 0, 1, 4, 4, 0, 4, 4, 1, 1, 4, 0, 1, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 0, 1, 4, 4, 1, 0, 1, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 1, 4, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 4, 4, 4, 4, 0, 0, 4, 1, 4, 4, 1, 4, 1, 4, 0, 4, 4, 0, 4, 1, 4, 4, 4, 4, 0, 0, 4, 1, 4, 4, 4, 1, 4, 4, 1, 0, 1, 0, 4, 4, 1, 1, 4, 0, 4, 0, 0, 1, 4, 4, 0, 4, 4, 4, 0, 4, 1, 0, 1, 0, 0, 1, 4, 4, 4, 4, 0, 1, 4, 4, 1, 0, 4, 4, 4, 1, 4, 4, 0, 0, 1, 4, 0, 4, 4, 4, 0, 1, 0, 1, 4, 0, 4, 4, 0, 4, 4, 4, 4, 1, 0, 0, 1, 1, 0, 0, 1, 4, 4, 1, 4, 0, 1, 1, 0, 4, 0, 4, 4, 0, 1, 4, 1, 1, 4, 4, 4, 4, 0, 4, 0, 1, 1, 1, 4, 0, 1, 0, 4, 4, 0, 1, 4, 0, 4, 0, 0, 4, 1, 1, 1, 1, 0, 4, 4, 4, 1, 0, 4, 4, 0, 1, 1, 1, 1, 1, 4, 0, 4, 1, 4, 1, 0, 1, 4, 0, 0, 0, 4, 0, 1, 4, 4, 4, 1, 0, 4, 4, 4, 0, 1, 4, 4, 1, 0, 4, 4, 0, 1, 4, 1, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 1, 4, 0, 4, 1, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 0, 4, 4, 1, 1, 0, 0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 1, 0, 4, 4, 4, 4, 4, 4, 0, 1, 4, 0, 4, 1, 1, 4, 1, 0, 0, 4, 0, 1, 0, 1, 1, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 4, 4, 1, 1, 1, 4, 1, 0, 1, 0, 4, 4, 1, 0, 4, 4, 4, 1, 1, 4, 4, 4, 0, 0, 4, 1, 0, 4, 1, 4, 4, 0, 4, 1, 1, 1, 0, 1, 1, 4, 4, 4, 4, 4, 4, 1, 0, 4, 4, 4, 0, 4, 0, 0, 0, 4, 1, 4, 4, 4, 4, 1, 1, 0, 4, 1, 4, 1, 4, 4, 4, 4, 1, 4, 4, 1, 0, 4, 1, 1, 4, 0, 0, 1, 4, 1, 0, 1, 4, 4, 4, 4, 1, 4, 4, 1, 4, 0, 1, 4, 0, 1, 4, 4, 1, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 1, 4, 1, 0, 1, 4, 4, 1, 1, 1, 4, 1, 4, 4, 4, 1, 1, 4, 1, 4, 4, 0, 1, 4, 1, 4, 4, 0, 4, 4, 1, 0, 0, 4, 4, 1, 1, 1, 4, 4, 1, 0, 0, 4, 4, 4, 1, 4, 0, 1, 1, 4, 4, 0, 4, 0, 1, 4, 1, 4, 4, 1, 1, 1, 1, 0, 1, 0, 1, 4, 1, 1, 4, 0, 0, 1, 0, 1, 4, 0, 4, 0, 4, 1, 1, 1, 4, 4, 1, 1, 0, 4, 1, 1, 4, 1, 4, 4, 4, 0, 1, 0, 4, 4, 0, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 1, 1, 4, 1, 4, 4, 1, 4, 0, 1, 1, 4, 1, 4, 4, 4, 4, 0, 4, 0, 4, 4, 1, 0, 4, 4, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 0, 0, 4, 4, 1, 0, 4, 1, 4, 1, 0, 0, 4, 4, 4, 1, 4, 0, 4, 4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 0, 1, 4, 1, 0, 4, 0, 4, 1, 4, 4, 4, 4, 1, 0, 4, 1, 4, 4, 4, 0, 1, 1, 0, 0, 4, 0, 4, 1, 1, 4, 4, 4, 0, 1, 0, 1, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 1, 4, 1, 0, 1, 4, 4, 0, 1, 0, 4, 0, 1, 4, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 1, 0, 1, 4, 0, 0, 4, 1, 1, 0, 4, 1, 4, 4, 4, 4, 4, 4, 0, 4, 4, 1, 4, 1, 1, 1, 4, 0, 4, 0, 4, 4, 0, 1, 1, 4, 1, 1, 1, 4, 1, 4, 4, 1, 1, 1, 4, 4, 1, 4, 1, 0, 0, 1, 4, 0, 4, 1, 4, 1, 1, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 0, 0, 0, 0, 0, 4, 1, 1, 4, 1, 4, 4, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 4, 4, 0, 1, 4, 0, 4, 4, 1, 0, 4, 1, 0, 4, 0, 0, 4, 0, 1, 0, 4, 0, 0, 4, 4, 4, 4, 0, 0, 0, 1, 4, 4, 0, 4, 1, 4, 0, 4, 1, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 4, 0, 1, 0, 1, 1, 1, 0, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 1, 1, 4, 4, 0, 0, 1, 4, 4, 4, 0, 1, 4, 4, 1, 1, 0, 1, 4, 1, 0, 4, 4, 1, 4, 1, 0, 1, 0, 4, 4, 1, 1, 4, 1, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 1, 4, 4, 0, 4, 1, 1, 1, 0, 4, 0, 1, 4, 4, 4, 0, 4, 1, 0, 0, 4, 4, 4, 1, 4, 1, 4, 0, 0, 4, 4, 4, 0, 4, 1, 4, 1, 4, 0, 4, 4, 0, 0, 4, 4, 0, 4, 4, 0, 4, 4, 1, 4, 0, 0, 4, 4, 0, 4, 4, 1, 1, 0, 0, 1, 0, 4, 4, 4, 1, 4, 4, 1, 0, 1, 4, 0, 0, 4, 0, 4, 4, 4, 4, 1, 0, 4, 4, 4, 1, 4, 0, 4, 1, 1, 1, 0, 4, 1, 1, 1, 0, 4, 4, 4, 4, 0, 1, 0, 0, 1, 1, 4, 1, 0, 0, 4, 1, 4, 1, 4, 4, 4, 0, 4, 1, 4, 0, 4, 4, 1, 4, 4, 1, 4, 1, 0, 4, 4, 1, 4, 0, 4, 4, 4, 0, 1, 0, 4, 1, 4, 1, 4, 4, 4, 0, 1, 1, 1, 4, 0, 0, 4, 4, 4, 1, 0, 4, 4, 1, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 1, 4, 1, 1, 1, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 4, 0, 0, 0, 4, 1, 4, 4, 4, 0, 0, 1, 0, 0, 4, 1, 0, 0, 1, 0, 4, 4, 0, 4, 4, 0, 4, 1, 0, 4, 4, 4, 4, 4, 1, 4, 1, 1, 0, 1, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 0, 0, 4, 1, 1, 4, 1, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 1, 0, 4, 4, 0, 4, 4, 1, 4, 4, 0, 4, 0, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 0, 4, 0, 1, 0, 1, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 0, 0, 4, 1, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 4, 1, 4, 4, 4, 4, 1, 4, 4, 1, 1, 1, 4, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 0, 4, 0, 4, 1, 1, 4, 0, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 0, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 0, 4, 0, 4, 4, 1, 1, 4, 0, 1, 4, 1, 4, 1, 4, 4, 0, 4, 4, 4, 1, 1, 0, 1, 4, 0, 1, 4, 4, 1, 4, 1, 1, 1, 0, 0, 0, 4, 1, 0, 0, 0, 1, 1, 4, 4, 1, 1, 4, 0, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 1, 0, 1, 0, 4, 4, 1, 4, 4, 0, 4, 4, 0, 0, 0, 1, 0, 1, 4, 1, 1, 0, 4, 1, 0, 0, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 1, 0, 4, 0, 1, 0, 4, 4, 4, 1, 4, 1, 0, 4, 4, 1, 4, 4, 4, 4, 1, 0, 1, 4, 1, 0, 1, 4, 0, 4, 1, 4, 1, 1, 0, 1, 0, 0, 0, 4, 4, 4, 0, 1, 0, 0, 0, 0, 4, 0, 4, 4, 1, 4, 1, 1, 0, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 0, 0, 1, 0, 4, 4, 1, 0, 0, 4, 1, 4, 4, 1, 4, 1, 0, 1, 4, 4, 0, 4, 1, 4, 1, 1, 0, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 4, 1, 0, 4, 4, 4, 1, 4, 4, 4, 4, 1, 0, 0, 4, 0, 4, 4, 4, 0, 0, 4, 1, 0, 1, 0, 1, 4, 4, 4, 0, 4, 0, 1, 4, 4, 4, 1, 1, 4, 0, 0, 1, 4, 0, 1, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 0, 0, 1, 0, 0, 0, 0, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 0, 0, 0, 1, 1, 4, 0, 4, 4, 4, 4, 0, 4, 1, 1, 4, 4, 4, 0, 0, 1, 1, 4, 1, 1, 0, 4, 0, 1, 4, 4, 1, 4, 1, 4, 1, 0, 0, 4, 4, 0, 1, 4, 1, 4, 4, 4, 0, 1, 1, 1, 4, 1, 4, 4, 4, 4, 0, 1, 4, 4, 4, 1, 4, 0, 0, 1, 4, 0, 0, 1, 1, 0, 4, 4, 4, 4, 1, 1, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4, 0, 4, 1, 4, 0, 4, 1, 0, 4, 1, 1, 0, 0, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 0, 4, 4, 0, 4, 4, 1, 4, 1, 4, 1, 1, 4, 4, 4, 4, 1, 4, 4, 4, 0, 4, 1, 0, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 0, 1, 1, 0, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 1, 0, 4, 0, 1, 4, 4, 4, 1, 1, 1, 4, 4, 4, 1, 0, 4, 4, 1, 4, 4, 4, 4, 1, 4, 0, 4, 0, 4, 4, 4, 4, 1, 4, 0, 0, 1, 0, 4, 0, 1, 0, 4, 4, 1, 4, 4, 4, 1, 1, 4, 4, 1, 1, 4, 1, 4, 1, 4, 4, 4, 1, 4, 1, 0, 4, 0, 0, 1, 0, 1, 4, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 1, 0, 4, 1, 0, 4, 4, 1, 0, 4, 4, 4, 1, 4, 0, 1, 4, 4, 4, 4, 0, 0, 4, 1, 1, 1, 0, 0, 1, 4, 1, 4, 1, 1, 4, 4, 0, 1, 0, 4, 1, 4, 4, 0, 0, 0, 4, 4, 0, 4, 4, 1, 1, 4, 4, 4, 1, 1, 0, 0, 0, 0, 4, 4, 1, 1, 1, 0, 1, 1, 4, 0, 1, 4, 0, 4, 4, 4, 4, 0, 0, 0, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 1, 4, 4, 4, 4, 0, 4, 4, 0, 1, 4, 0, 1, 4, 0, 0, 0, 1, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 1, 4, 4, 1, 4, 4, 0, 1, 4, 1, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 1, 4, 4, 1, 0, 1, 0, 4, 1, 4, 1, 0, 1, 4, 4, 4, 0, 4, 1, 0, 0, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 4, 0, 1, 1, 0, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 1, 4, 4, 0, 1, 4, 0, 0, 4, 0, 4, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 0, 0, 4, 1, 4, 4, 0, 1, 4, 1, 4, 0, 4, 1, 4, 4, 4, 1, 4, 1, 4, 0, 1, 1, 1, 0, 0, 1, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 0, 0, 4, 1, 4, 1, 0, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 0, 4, 0, 1, 1, 0, 0, 4, 0, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 1, 1, 4, 4, 0, 1, 4, 4, 0, 4, 1, 1, 4, 0, 4, 4, 4, 4, 1, 4, 4, 4, 0, 4, 0, 4, 1, 4, 4, 0, 4, 4, 4, 1, 4, 4, 1, 1, 1, 1, 4, 4, 4, 1, 0, 0, 4, 4, 1, 1, 4, 0, 4, 4, 4, 0, 4, 1, 4, 4, 4, 4, 0, 1, 4, 0, 0, 4, 4, 4, 0, 4, 4, 1, 4, 4, 4, 4, 4, 4, 0, 4, 1, 0, 1, 4, 4, 4, 4, 4, 0, 0, 4, 0, 1, 4, 4, 0, 0, 0, 4, 0, 1, 4, 1, 4, 1, 4, 0, 1, 4, 4, 4, 1, 0, 4, 0, 1, 0, 1, 4, 4, 0, 4, 4, 0, 1, 0, 4, 4, 0, 4, 0, 1, 0, 1, 1, 1, 0, 4, 0, 0, 4, 0, 0, 1, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 1, 4, 4, 0, 4, 4, 0, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 4, 1, 1, 4, 1, 0, 4, 1, 0, 4, 4, 0, 0, 1, 1, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 0, 0, 1, 4, 4, 1, 4, 1, 4, 4, 4, 4, 1, 1, 4, 1, 1, 1, 4, 1, 4, 0, 1, 4, 4, 4, 1, 4, 4, 4, 4, 1, 1, 4, 4, 4, 0, 4, 0, 0, 1, 4, 0, 0, 4, 1, 4, 0, 4, 1, 1, 4, 4, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 4, 4, 1, 0, 1, 4, 4, 1, 4, 0, 4, 4, 0, 4, 4, 1, 4, 4, 0, 4, 1, 1, 4, 4, 0, 0, 0, 4, 1, 0, 4, 1, 1, 1, 0, 0, 0, 4, 4, 1, 0, 1, 1, 4, 4, 4, 1, 0, 4, 1, 0, 1, 0, 4, 1, 0, 4, 4, 1, 4, 1, 4, 4, 0, 4, 0, 1, 4, 1, 1, 4, 4, 1, 0, 1, 4, 0, 4, 4, 0, 0, 0, 1, 1, 4, 1, 0, 4, 1, 4, 4, 0, 1, 4, 4, 0, 1, 4, 4, 4, 4, 1, 1, 4, 0, 4, 4, 1, 4, 1, 4, 4, 4, 0, 4, 4, 0, 4, 4, 1, 4, 0, 4, 4, 0, 4, 4, 0, 1, 4, 4, 0, 0, 0, 1, 4, 0, 4, 1, 4, 4, 4, 1, 4, 4, 0, 0, 0, 0, 1, 4, 0, 4, 4, 1, 0, 4, 4, 0, 4, 4, 1, 4, 0, 0, 0, 4, 4, 4, 1, 1, 0, 0, 4, 0, 0, 1, 4, 0, 0, 4, 4, 4, 0, 4, 4, 1, 4, 1, 1, 0, 1, 0, 4, 4, 4, 4, 1, 4, 0, 0, 4, 1, 4, 0, 0, 1, 0, 4, 4, 1, 0, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4, 0, 0, 0, 4, 4, 0, 1, 4, 4, 4, 4, 1, 4, 0, 4, 0, 0, 1, 1, 1, 4, 1, 0, 4, 4, 4, 1, 0, 4, 1, 4, 4, 1, 4, 1, 1, 0, 4, 0, 4, 4, 4, 1, 4, 0, 4, 4, 0, 1, 0, 0, 1, 4, 0, 1, 4, 4, 0, 0, 0, 1, 0, 4, 4, 4, 0, 1, 4, 4, 1, 0, 4, 1, 4, 1, 0, 0, 0, 0, 4, 0, 1, 1, 4, 1, 1, 4, 4, 0, 1, 1, 4, 0, 1, 0, 1, 4, 4, 0, 4, 0, 0, 4, 1, 0, 4, 4, 4, 0, 4, 1, 4, 1, 0, 4, 4, 0, 4, 4, 1, 4, 4, 0, 1, 4, 4, 0, 4, 4, 0, 1, 0, 4, 4, 1, 4, 0, 4, 0, 4, 4, 1, 1, 4, 0, 4, 0, 0, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1, 4, 0, 1, 4, 0, 0, 4, 4, 1, 1, 0, 4, 0, 4, 0, 1, 4, 0, 1, 0, 0, 0, 0, 4, 4, 4, 1, 4, 1, 0, 4, 1, 0, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 0, 0, 1, 4, 1, 1, 0, 1, 1, 0, 1, 4, 4, 4, 4, 4, 0, 4, 1, 0, 1, 4, 0, 1, 0, 0, 4, 0, 0, 0, 0, 4, 1, 0, 0, 1, 4, 1, 4, 4, 4, 1, 0, 0, 4, 0, 4, 0, 4, 4, 0, 0, 1, 1, 4, 4, 4, 1, 1, 4, 0, 1, 0, 0, 1, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 0, 0, 1, 1, 1, 1, 0, 4, 4, 4, 0, 1, 4, 1, 1, 4, 1, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 0, 1, 1, 0, 4, 4, 4, 0, 4, 0, 1, 0, 4, 1, 4, 0, 1, 4, 4, 1, 0, 4, 0, 4, 0, 1, 0, 4, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 0, 4, 4, 4, 4, 0, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 0, 1, 0, 4, 0, 4, 1, 1, 4, 0, 4, 0, 1, 1, 1, 1, 4, 0, 4, 0, 0, 0, 4, 4, 1, 4, 0, 1, 1, 1, 4, 4, 0, 4, 0, 1, 1, 4, 0, 1, 1, 4, 4, 4, 1, 4, 1, 4, 0, 1, 0, 0, 1, 1, 1, 1, 0, 4, 4, 0, 4, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 0, 4, 4, 0, 1, 0, 4, 1, 0, 4, 4, 0, 4, 0, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 0, 0, 4, 4, 1, 4, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 4, 1, 0, 0, 4, 0, 0, 4, 1, 1, 0, 0, 4, 1, 0, 0, 4, 0, 4, 1, 0, 4, 1, 0, 4, 1, 4, 4, 4, 1, 0, 1, 4, 1, 1, 4, 0, 4, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 1, 0, 1, 0, 4, 4, 4, 1, 4, 4, 4, 1, 0, 0, 4, 4, 0, 0, 1, 4, 1, 0, 0, 0, 4, 0, 4, 1, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 0, 4, 0, 0, 4, 4, 0, 4, 1, 1, 4, 4, 1, 0, 0, 1, 0, 4, 4, 0, 4, 4, 4, 0, 0, 4, 1, 4, 4, 0, 4, 4, 1, 0, 0, 4, 1, 1, 0, 1, 4, 4, 1, 4, 4, 1, 4, 1, 1, 4, 0, 0, 0, 4, 4, 4, 4, 4, 1, 1, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 4, 1, 0, 4, 4, 0, 4, 1, 0, 4, 4, 4, 0, 4, 0, 4, 1, 4, 4, 4, 4, 4, 0, 0, 1, 4, 0, 1, 0, 0, 0, 0, 4, 0, 4, 4, 4, 1, 0, 4, 0, 4, 4, 4, 0, 4, 1, 1, 4, 4, 0, 4, 1, 0, 4, 1, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 1, 0, 1, 4, 4, 1, 4, 1, 4, 1, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 1, 1, 4, 4, 4, 1, 4, 0, 4, 4, 0, 1, 0, 4, 0, 4, 4, 1, 0, 0, 1, 1, 4, 0, 4, 4, 4, 1, 0, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 0, 1, 0, 4, 4, 1, 0, 4, 4, 0, 0, 1, 4, 4, 4, 4, 4, 4, 0, 0, 4, 1, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 1, 0, 4, 0, 1, 4, 0, 1, 1, 4, 1, 4, 0, 4, 0, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 0, 1, 4, 4, 0, 1, 0, 1, 1, 4, 0, 4, 4, 4, 1, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 1, 4, 4, 4, 4, 1, 0, 4, 4, 4, 1, 1, 4, 1, 4, 0, 4, 0, 4, 1, 0, 0, 4, 0, 1, 4, 0, 0, 4, 1, 4, 4, 0, 0, 4, 4, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 1, 4, 1, 0, 4, 0, 0, 4, 0, 4, 4, 4, 4, 4, 1, 4, 0, 1, 0, 0, 4, 4, 1, 0, 4, 4, 4, 1, 0, 0, 4, 4, 4, 0, 4, 0, 1, 1, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 1, 1, 4, 0, 4, 4, 1, 0, 0, 4, 4, 1, 1, 1, 4, 1, 4, 1, 1, 4, 1, 0, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 0, 4, 1, 4, 0, 4, 4, 1, 4, 1, 0, 4, 1, 0, 4, 1, 4, 0, 0, 4, 0, 4, 4, 1, 4, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 4, 4, 4, 1, 0, 4, 4, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 4, 0, 4, 4, 4, 0, 1, 4, 0, 4, 1, 4, 0, 0, 4, 1, 0, 1, 1, 1, 0, 0, 0, 1, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 1, 1, 1, 1, 0, 4, 0, 4, 0, 1, 0, 0, 0, 4, 4, 0, 4, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 0, 4, 0, 1, 1, 4, 0, 4, 4, 4, 0, 0, 1, 0, 4, 1, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 1, 4, 4, 0, 4, 0, 4, 4, 0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 4, 0, 4, 1, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 1, 4, 4, 4, 0, 4, 1, 4, 1, 0, 0, 0, 4, 1, 1, 4, 1, 0, 0, 0, 1, 0, 4, 1, 4, 4, 1, 4, 4, 4, 0, 0, 0, 4, 4, 0, 0, 1, 0, 4, 4, 0, 1, 4, 4, 4, 1, 4, 4, 4, 0, 1, 4, 4, 1, 4, 1, 4, 4, 4, 0, 0, 4, 4, 1, 1, 1, 1, 4, 1, 1, 0, 4, 1, 0, 0, 4, 1, 4, 4, 4, 0, 4, 0, 1, 1, 0, 4, 4, 4, 1, 4, 4, 1, 0, 4, 4, 4, 4, 0, 0, 1, 4, 4, 4, 4, 1, 1, 4, 0, 0, 1, 0, 4, 0, 1, 0, 4, 0, 4, 4, 4, 4, 0, 1, 0, 4, 0, 4, 4, 4, 1, 4, 0, 1, 0, 4, 0, 4, 4, 1, 1, 0, 1, 0, 1, 4, 4, 0, 0, 1, 1, 4, 4, 4, 4, 1, 4, 4, 0, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 0, 1, 0, 4, 1, 0, 4, 4, 1, 1, 1, 4, 0, 1, 1, 4, 0, 4, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 4, 4, 0, 4, 1, 1, 4, 0, 0, 4, 4, 1, 1, 1, 4, 4, 0, 0, 4, 4, 4, 4, 1, 4, 0, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 1, 0, 1, 0, 1, 4, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 4, 4, 4, 4, 1, 0, 0, 4, 0, 4, 4, 4, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 4, 4, 1, 4, 0, 0, 4, 4, 4, 0, 0, 4, 1, 4, 4, 1, 1, 0, 0, 1, 0, 4, 1, 0, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 0, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 0, 1, 1, 4, 1, 4, 4, 4, 1, 0, 4, 0, 4, 1, 0, 0, 1, 0, 4, 0, 4, 0, 1, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 4, 0, 0, 4, 4, 1, 1, 4, 4, 0, 1, 1, 4, 1, 0, 1, 0, 4, 0, 1, 4, 1, 0, 1, 1, 4, 4, 1, 0, 4, 1, 1, 4, 1, 4, 1, 1, 1, 4, 4, 4, 0, 4, 0, 4, 4, 1, 0, 0, 1, 4, 4, 4, 0, 4, 0, 0, 1, 4, 4, 4, 4, 0, 0, 1, 1, 4, 0, 4, 0, 4, 4, 1, 1, 1, 4, 4, 1, 4, 4, 1, 0, 4, 4, 4, 4, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 4, 1, 0, 4, 0, 4, 4, 4, 4, 0, 0, 0, 4, 0, 1, 4, 0, 4, 4, 0, 1, 4, 4, 0, 1, 1, 0, 0, 0, 1, 0, 4, 4, 1, 0, 1, 0, 4, 4, 4, 0, 4, 1, 1, 0, 4, 0, 4, 0, 4, 4, 0, 0, 4, 0, 1, 4, 1, 4, 4, 0, 4, 4, 1, 1, 4, 1, 4, 1, 0, 1, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 0, 1, 1, 4, 1]

    # -- merge
    # print(f"rand_score {metrics.adjusted_rand_score(ander, labels)}")
    # print(f"mutual_info_score {metrics.adjusted_mutual_info_score(ander, labels)}")
    # # print(f"silhouette_score {metrics.silhouette_score(df, labels)}")   # bad
    # print(f"calinski_harabasz_score {metrics.calinski_harabasz_score(df, labels)}")
    # print(f"davies_bouldin_score {metrics.davies_bouldin_score(df, labels)}")
    #
    # print(f"contingency_matrix {metrics.cluster.contingency_matrix(ander, labels)}")
    # labels = ac.fit_predict(df)
    # -- plot
    pca = KernelPCA(
        # kernel='rbf',
        n_components=2)
    X_principal2 = pca.fit_transform(df_o)
    X_principal2 = pd.DataFrame(X_principal2, columns=['P1', 'P2'])
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(X_principal2['P1'], X_principal2['P2'], c=labels, cmap='rainbow', s=20)
    t = f"Кластеризация на PCA"
    plt.title(t)
    cl = [str(x) for x in np.unique(labels)]
    plt.legend(handles=scatter.legend_elements()[0], labels=cl)
    plt.savefig(f'{i}pca.png')
    plt.show()
    plt.close()

    return labels


def select_and_set_types(p1: str, p2: str) -> (str, str):
    # -- SELECT COLUMNS
    cols2 = [
        'CLIENT_AGE',
        'CLIENT_GENDER',
        'CLIENT_MARITAL_STATUS',
        'CLIENT_DEPENDENTS_COUNT',
        'CLIENT_WI_EXPERIENCE',
        'ANKETA_SCORING',
        'OKB_SCORING',
        'EQUIFAX_SCORING',
        'NBKI_SCORING',
        'AUTO_DEAL_INITIAL_FEE',
        'OKB_RATING_SCORING',  # KI
        'NBKI_RATING_SCORING',  # KI
        'EQUIFAX_RATING_SCORING',
        'МБКИ_треб_адрес', 'МБКИ_треб_исп_пр',
        'МБКИ_треб_пассп', 'МБКИ_требаналотч', 'МБКИ_нет_огр', 'МБКИ_недост',
        'МБКИ_розыск', 'МБКИ_невыполнена', 'МБКИ_налспецуч', 'МБКИ_налсуд',
        'МБКИ_кат_и_срок', 'МБКИ_данные_не',
        # 'Час создания заявки',
        # 'Месяц создания заявки',
        'День недели',
        'ander',
        # -- Scorings

        'nbki_count_total', 'nbki_count_acc_without_credit_cards', 'nbki_count_acc_credit_cards',
        'nbki_count_acc_ipoteka', 'nbki_count_acc_lower50_closed', 'nbki_count_acc_higher50_closed',
        'nbki_acc_payAsAgreed', 'nbki_acc_late29Days', 'nbki_acc_payAsAgreed_without_cc',
        'nbki_acc_late29Days_without_cc', 'equifax_count_total', 'equifax_count_acc_without_credit_cards',
        'equifax_count_acc_credit_cards', 'equifax_count_acc_ipoteka', 'equifax_count_acc_lower50_closed',
        'equifax_count_acc_higher50_closed', 'equifax_acc_payAsAgreed', 'equifax_acc_late29Days',
        'equifax_acc_payAsAgreed_without_cc', 'equifax_acc_late29Days_without_cc', 'okb_count_total',
        'okb_count_acc_without_credit_cards', 'okb_count_acc_credit_cards', 'okb_count_acc_ipoteka',
        'okb_count_acc_lower50_closed', 'okb_count_acc_higher50_closed', 'okb_acc_payAsAgreed', 'okb_acc_late29Days',
        'okb_acc_payAsAgreed_without_cc', 'okb_acc_late29Days_without_cc', 'all_count_total',
        'all_count_acc_without_credit_cards', 'all_count_acc_credit_cards', 'all_count_acc_ipoteka',
        'all_count_acc_lower50_closed', 'all_count_acc_higher50_closed', 'all_acc_payAsAgreed', 'all_acc_late29Days',
        'all_acc_payAsAgreed_without_cc', 'all_acc_late29Days_without_cc', 'mbki_isMassLeader', 'mbki_isMassAddress',
        'mbki_inBanksStopLists', 'mbki_isInnConfirmed', 'mbki_informationNotFound', 'mbki_hasAdministrativeCrimesData',
        #
        '0prov_status_code',
        '0prov_status_type',
        '0prov_status_groupid',
        '0prov_status_isActing',
        '0prov_status_groupName',
        '0prov_companyType',
        '0prov_okfs_code',
        '0prov_includeInList',
        '0prov_change_history',
        '0prov_legalAddresses',
        'ex4scor_scoring',
        'afsnbki_scoring',
        'okbnh_MA_SMT',
        'okbnh_MA_EMP',
        'okbnh_MA_REF',
        'okbnh_MA_MTE',
        'okbnh_MA_SPA',
        'okbnh_MA_SPE',
        'okbnh_MULT_M',
        'okbnh_MA_MS',
        'okbnh_MA_SAM',
        'okbnh_MA_AS_',
        'okbnh_LCL_MA', 'okbnh_MA_MS_',
        'okbnh_MA_RAD',
        'okbnh_MA_PAS', 'okbnh_MA_SWT', 'okbnh_MA_AS',
        'nbki_biom_resp_matchImages',
        'nbki_biom_resp_matchResults',
        'nbki_biom_resp_match_max',
        # 'nbki_biom_resp_match_avg', # corr with nbki_biom_resp_match_max
        'INITIAL_FEE_DIV_CAR_PRICE',
        'AUTO_DEAL_COST_REQUESTED'
        
        # -- tmp
        'TT',
    ]

    # cols2=[]

    df_train: pd.DataFrame = pd.read_pickle(p1)
    df_test: pd.DataFrame = pd.read_pickle(p2)

    df_train.drop(columns=['080.4', '091', 'has_codes', "DEAL_CREATED_DATE"], inplace=True)

    df_train = df_train[cols2 + ['id']]
    df_test = df_test[cols2 + ['id']]

    # -- remove 1 value columns
    df_train, dropped_cols = remove_single_unique_values(df_train)
    df_test.drop(columns=dropped_cols, inplace=True)

    # -- bool to int
    # we assume that Bool can have [None, True, False, class pandas._libs.missing.NAType] values
    for c in df_train.columns:
        uv = df_train[c].unique()
        if len(uv) <= 4 and all(
                [isinstance(x, pd._libs.missing.NAType) or x in [None, True, False] for x in uv]):  # detect boolean
            print(c, uv)
            df_train[c] = pd.to_numeric(df_train[c].replace({True: 1, False: 0}), errors='coerce')
            df_test[c] = pd.to_numeric(df_test[c].replace({True: 1, False: 0}), errors='coerce')

    # -- detect '0.1234' columns and convert to Int64
    # we assume that object column is numerical if we have >3 numbers in values
    categorial_columns = df_train.select_dtypes(exclude=["number"]).columns.tolist()
    r = re.compile(r'^[+-]?(\d{1,16}(\.\d*)?|\.\d{1,16})([eE][+-]?\d+)?$')
    for c in categorial_columns:
        s_isn: pd.Series = df_train[c].apply(lambda x: bool(r.match(x) if isinstance(x, str) else False))
        co: pd.Series = s_isn.value_counts()
        if True in co.index.tolist() and co.loc[True] > 3:  # we have numerical
            df_train[c] = pd.to_numeric((df_train[c]), errors='coerce')
            df_test[c] = pd.to_numeric((df_test[c]), errors='coerce')

    p1 = 'select_train.pickle'
    p2 = 'select_test.pickle'
    save(p1, df_train)
    save(p2, df_test)
    return p1, p2


def split(p: str, t1: str, t2: str, split_date=None) -> (str, str):
    """
    :param p: source dataframe
    :param t1: train dataframe
    :param t2: test dataframe
    :param split_date:
    :return:
    """
    df: pd.DataFrame = pd.read_pickle(p)

    X_train, X_test, y_train, y_test = train_test_split(df, df['ander'], test_size=0.20, shuffle=False,
                                                        random_state=2, stratify=None)
    print(X_train.shape, X_test.shape)
    print("last date train:", X_train['DEAL_CREATED_DATE'].sort_values().tail(1))
    print("last date test:", X_test['DEAL_CREATED_DATE'].sort_values().tail(1))
    if split_date is None:
        split_date: pd.Timestamp = X_train['DEAL_CREATED_DATE'].sort_values().tail(1).to_list()[0]
    # -- test ids
    ids: list = pd.read_pickle('id.pickle')
    print("ids check:")
    assert all(df['id'] == ids)
    print("SPLIT date =", str(split_date))
    df_train = df[df['DEAL_CREATED_DATE'] <= split_date].copy()
    df_test = df[df['DEAL_CREATED_DATE'] > split_date].copy()

    # del date col
    rj_codes_not_about_client = ['080.2', '080.4', '091', '090.3', '090.3.2', '090.5', '090.10', '090.22', '080.3',
                                 '010.2.4']  # , '097'

    # save rejected codes bar
    codes = []

    def b(x):
        if any([y in str(x).split('; \\n') for y in codes]):
            return 1
        else:
            return 0

    for c in rj_codes_not_about_client:
        codes = [c]
        df_train[c] = df_train['DC_REJECTED_CODES'].apply(b)
    codes_rej = df_train[df_train['ander'] == 0][rj_codes_not_about_client].sum()
    codes_appr = df_train[df_train['ander'] == 1][rj_codes_not_about_client].sum()
    pd.DataFrame({"отклоненные": codes_rej, "одобренные": codes_appr}).plot.bar()
    plt.savefig('bad_rj_codes_in_train')
    plt.close()

    # -- remove bad rj codes from train
    def a(x):
        if any([y in str(x).split('; \\n') for y in rj_codes_not_about_client]):
            return 1
        else:
            return 0

    df_train['has_codes'] = df_train['DC_REJECTED_CODES'].apply(a)

    df_train_rej = df_train[df_train['ander'] == 0]
    # df_train_rej = df_train
    df_train_rej = df_train_rej[df_train_rej['has_codes'] == 1]

    rj_codes_not_about_client = ['097']
    df_train['has_codes'] = df_train['DC_REJECTED_CODES'].apply(a)
    df_train_appr = df_train[df_train['ander'] == 1]
    df_train_appr = df_train_appr[df_train_appr['has_codes'] == 1]
    # print(df_train['mbki_isMassAddress'].unique())
    # print(df_train['mbki_isMassPhone'].unique())
    # exit()
    print("dropped with reject codes: " + str(rj_codes_not_about_client))
    print(df_train_rej.shape[0])
    # -- drop codes
    df_train = df_train.drop(index=df_train_rej.index)
    df_train = df_train.drop(index=df_train_appr.index)

    # print(df_train.shape)
    # exit()

    save('id_train.pickle', df_train['id'].tolist())
    save('id_test.pickle', df_test['id'].tolist())

    save(t1, df_train)
    save(t2, df_test)
    return t1, t2
