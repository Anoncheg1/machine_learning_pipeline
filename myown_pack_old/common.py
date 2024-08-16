import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def print_list(l:list):
    if type(l[0]) == str:
        format_row = "{:<25} " * (len(l) + 1)
        return format_row.format("", *l).strip()
    elif type(l[0]) == str:
        format_row = "{:<25} " * (len(l[0]) + 1)
        return [format_row.format("", *x).strip() for x in l]
    else:
        print("ERROR! in print_list!!!!")


def closest(alph: iter, source: list) -> list:
    """
    alph = [1, 2, 5, 7]
    source = [1, 2, 3, 6]  # 3, 6 replace to closest
    return [1,2,2,5]
    """

    target = source[:]
    for i, s in enumerate(source):
        if s not in alph:
            distance = [(abs(x - s), x) for x in alph]
            res = min(distance, key=lambda x: x[0])
            target[i] = res[1]
    return target


def impute(df_o: pd.DataFrame, max_iter=7, percent=0.3, exclude: list = None) -> pd.DataFrame:
    """
    All numeric columns myst be int32 or int64 or int
    1) Separate categorical, numerical and bad_numerical columns
    2) Encode categorical columns (dummy a,b - 0,1)
    3) Concat encoded and numerical columns
    3) predict missing values (imputation)
    4) decode categorical columns
    """
    from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm
    # -- object ot str type
    # categorial_columns2 = df_o.select_dtypes(include=["object"]).columns
    # for c in categorial_columns2:
    #     df_o[c] = df_o[c].astype(str)
    # print(df_o.isna().sum())
    # print(df_o.columns)
    # return

    # COLUMNS WITH COUNT(NaN) > 50% - just saved and will be imputed, then replaced to old
    exception_columns = []
    for c in df_o:
        if df_o[c].hasnans:
            if df_o[c].isna().sum() > (df_o.shape[0]*percent):  # NaN > 50%
                exception_columns.append(c)
            else:
                print(c)
    df_exception = df_o[exclude + exception_columns].copy()

    # SAVE COLUMN TYPES
    columns_dtypes = {x[0]: x[1] for x in zip(df_o.columns, df_o.dtypes)}
    # SEPARATE CATEGORICAL AND NUMERICAL COLUMNS
    # {'int64', 'object', 'float64'}
    categorical_columns = []
    numerical_columns = []
    numerical_columns_excluded = []
    for c in df_o:  # columns
        if df_o[c].dtype.name == 'object':
            categorical_columns.append(c)
            # print(c, df_o[c].dtype)
        # elif c in exception_columns:
        #     numerical_columns_excluded.append(c)
        else:
            numerical_columns.append(c)
            # SIDE EFFECT: replace Int32 to Object
            t = df_o[c].dtype.name
            # print(c, df_o[c].dtype.name)
            df_o[c].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_o[c] = df_o[c].astype(t)

    c_df: pd.DataFrame = df_o[categorical_columns].copy()
    n_df = df_o[numerical_columns].copy()
    n_e_df = df_o[numerical_columns_excluded].copy()
    print("c2", c_df.shape, n_df.shape, n_e_df.shape)

    # IMPUTE CATEGORICAL COLUMNS (temp)
    # c_df.replace(None, np.nan, inplace=True)
    # imp = SimpleImputer(strategy="most_frequent")
    # c_df = imp.fit_transform(c_df.astype('category'))
    # print(c_df.shape)
    # c_df = pd.DataFrame(c_df)
    # print("categorical has NaNs", c_df.isna().any().any())
    # # print(c_df.head(10).to_string())
    # print(c_df.shape)
    # exit()

    # print(df_a[df_a['Оценка кредитной истории Эквифакс'] == 'КИ отсутствует'].shape[0])
    # ENCODE CATEGORICAL COLUMNS
    label_encoders = {}
    c_df.fillna(value='NoneMy', inplace=True)
    for c in c_df:  # columns
        # save encoder
        le: LabelEncoder = LabelEncoder().fit(c_df[c])
        label_encoders[c] = le

        index = np.where(le.classes_ == 'NoneMy')[0]
        # encode
        c_df[c] = pd.Series(le.transform(c_df[c]))

        # replace NoneMy to np.NaN
        if index.shape[0] == 1 and c not in exception_columns:
            c_df[c].replace(index[0], np.nan, inplace=True)

    # df_a = c_df
    # print(df_a[df_a['Оценка кредитной истории Эквифакс'] == 1].shape[0])
    # return

    # CONCAN NUMERICAL AND ENCODED CATECOGORICAL
    df = pd.concat([n_df, c_df], axis=1, sort=False, verify_integrity=True)
    # print(df.shape)
    # return

    # IMPUTATION
    # from sklearn.ensemble import ExtraTreesRegressor
    # ExtraTreesRegressor()
    imp = IterativeImputer(random_state=0, verbose=2, initial_strategy='median',
                           max_iter=max_iter, n_nearest_features=None, sample_posterior=True)
    # TEMPORAL REPLACEMENT
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # df = df.to_numpy()
    for c in numerical_columns:
        if n_df[c].dtype.name == 'object':
            print(c, n_df[c].dtype.name)  # required
            raise Exception

    # print(df.to_numpy().shape)
    df: np.array = imp.fit_transform(df)
    # print(df.shape)
    # RESTORE NUMERICAL COLUMN TYPES and NAMES
    df = pd.DataFrame(df)  # numpy to pandas

    df.columns = numerical_columns + categorical_columns  # restore column names
    for c in tqdm(df.columns):  # columns
        if c in numerical_columns:
            df[c] = df[c].round().astype(columns_dtypes[c])  # float or int
        else:  # categorical
            alph = label_encoders[c].transform(label_encoders[c].classes_)  # original set without NaN
            if len(alph) != 0:
                df[c] = pd.Series(closest(alph, df[c])).astype(int)  # replace not original to closest original
            else:
                print("Error len(alph) == 0", alph)
    # print("c5", df['autocredit_car_info.`condition`'].tail(10).to_string())
    # Add numerical columns with NaN > 50%
    df = df.join(n_e_df)

    # df.reset_index(drop=True, inplace=True)
    # DECODE CATEGORICAL COLUMNS
    for c in categorical_columns:
        df[c] = label_encoders[c].inverse_transform(df[c])

    df = df.drop(df_exception.columns, axis=1)
    df = pd.concat([df, df_exception], axis=1, sort=False, verify_integrity=True)

    return df


def save(sp: str, obj:any) -> str:
    """
    save to .pickle
    :param sp: must be *.pickle
    :param obj: sp
    :return:
    """
    pd.to_pickle(obj, sp)
    print()
    if type(obj) == np.ndarray:
        print('-- ok --', sp, obj.shape)
    elif type(obj) == pd.DataFrame:
        print('-- ok --', sp, obj.shape, obj.columns.tolist())
    else:
        print('-- ok --', sp)
    return sp


def load(p, nrows:int=None) -> pd.DataFrame:
    """ if p:
    dataframe -> DataFrame
    .pickle -> DataFrame
    .csv -> DataFrame

    nrows = number of lines to read"""
    if type(p) == str:
        p: str = p
        if p.endswith('.csv'):
            df = pd.read_csv(p, index_col=0, low_memory=False, nrows=nrows)
        elif p.endswith('.pickle'):
            df: pd.DataFrame = pd.read_pickle(p)
        else:
            logging.error("load error p:", p)

    elif type(p) == pd.DataFrame:
        df: pd.DataFrame = p
    else:
        raise Exception("p is unknewn")
    return df


def check_id(df, id_path: str):
    ids: list = pd.read_pickle(id_path)
    print("ids check:", len(ids), df.shape[0])
    assert all(df['id'] == ids)


def rename_columns(p_or_df: pd.DataFrame or str, columns: dict, length=8) -> pd.DataFrame or str:
    """
    fields_dict = {'SRC_BANK': 'Наименование источника',
               'APP_SRC_REF': 'ID источника',
               'DEAL_ID': 'ID сделки ХД',
               'CLIENT_ID': 'ID клиента ХД"',
               'DEAL_SRC_REF': 'ID сделки ЦФТ'}
    ex.:
    new_columns = {x: str(i) for i, x in enumerate(df.columns.tolist())}
    del new_columns['id']
    df = rename_columns(df, new_columns)

    :param p_or_df:
    :param columns: old_column: new_column
    :param len: max length of new column
    :return:
    """
    if isinstance(p_or_df, str):
        df = load(p_or_df)
    else:
        df = p_or_df
    # -- make columns shorter
    new_columns = {k: v[:length] for k, v in columns.items()}
    # -- rename
    df.rename(columns=new_columns, inplace=True)
    # -- save
    if isinstance(p_or_df, str):
        return save('renamed.pickle', df)
    else:
        return df


def impute_v(p, max_iter=7, percent=0.3, exclude: list = None, remove: list = None):
    df: pd.DataFrame = pd.read_pickle(p)
    # -- drop columns
    df = df.drop(remove, 1)
    # -- drop rows
    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    # -- reset index
    df = df.reset_index(drop=True)
    # -- impute
    df = impute(df, max_iter=max_iter, percent=percent, exclude=exclude)

    # df_a = df
    # print(df_a[df_a['Оценка кредитной истории Эквифакс'] == 'КИ отсутствует'].shape[0])

    # correct na values
    # df2['Коды отказа'] = df2['Коды отказа'].apply(lambda x: x if x >= 0 else 0)
    return save('after_imputer.pickle', df)


def outliers_numerical(p, d=0.0001, ignore_columns: list = None, target=None):
    # from scipy.stats import skew
    df: pd.DataFrame = pd.read_pickle(p)
    if ignore_columns is None:
        ignore_columns = []
    ignore_columns += ['id']

    numerical_columns = df.select_dtypes(exclude=["object"]).columns

    tc = 0
    deleted = []  # (column_name, quantity)
    deleted_targ0 = []  # (column_name, quantity)
    deleted_targ1 = []  # (column_name, quantity)
    for c in numerical_columns:
        # print(c, df[c].unique())
        if ignore_columns is not None and c in ignore_columns:
            continue
        q_low = df[c].quantile(d)  # 0.0001
        q_hi = df[c].quantile(1-d)  # 0.9999
        # print(q_low, q_hi)

        df_filtered = df[(df[c] > q_hi) | (df[c] < q_low)]
        # if c == c_interest: # test
        #     print("filtered")
        #     print(df_filtered[c_interest].describe())
        #     print(df_filtered[c_interest].unique())
            # return
        deleted.append((c,df_filtered[c].shape[0]))
        if target is not None:
            deleted_targ0.append((c + '_0', df_filtered[df_filtered[target] == 0][c].shape[0]))
            deleted_targ1.append((c + '_1', df_filtered[df_filtered[target] == 1][c].shape[0]))

        tc += df_filtered[c].shape[0]
        df.drop(df_filtered.index, inplace=True)

    df_deleted = pd.DataFrame(deleted).set_index(0).sort_values(by=[1], ascending=False)
    # -- print report
    if not target:
        print(df_deleted.to_string())
    else:
        df_deleted0 = pd.DataFrame(deleted_targ0).set_index(0).sort_values(by=[1], ascending=False)
        df_deleted1 = pd.DataFrame(deleted_targ1).set_index(0).sort_values(by=[1], ascending=False)
        print(f"per target 0: {df_deleted0[1].sum()} , per target 1: {df_deleted1[1].sum()}")
        print(df_deleted0.to_string())
        print(df_deleted1.to_string())
    print()
    # -- remove
    df_deleted = df_deleted[df_deleted[1] > 0]
    # test
    # print(df[c_interest].unique())
    # print(df[c_interest].describe())
    # -- ids chech
    # ids: list = pd.read_pickle('id.pickle')
    # print("ids check:", all(df['id'] == ids))
    # -- save ids
    save('id_train.pickle', df['id'].tolist())

    print("filtered:", df_deleted)
    print("total filtered count:", tc)
    return save('without_outliers.pickle', df)


def condense_category(col: pd.Series, min_freq=1, new_name='other'):
    """ sparse classes lower min_freq percent replaced with new_name
    multicolumn: from chepelev_pack.common import sparse_classes"""
    vc_s = pd.value_counts(col)
    vcp_s = ((vc_s / vc_s.sum()) * 100)
    mask: pd.Series = vcp_s[vcp_s < min_freq]
    print(col.name, 'count of low frequent values:', mask.count(), '\t\tcount of saved values:', vcp_s[vcp_s >= min_freq].count())
    values_to_replace = {x: new_name for x in list(mask.index)}
    return col.replace(values_to_replace)


def condense_category_horizontal(source_series: pd.Series, sep: str=';', min_freq=1) -> pd.DataFrame:
    """
    convert rows of '9;25;26;43;44'
    to columns ['25', '26', 'others', 'np.NAN'] with values:
    [1, 1, 3, 0],
    where '25', '26' - 1, 1 are most frequent in all rows,
    3 - others, 0 - np.NAN

    :param source_series: source row with long random strings separated by sep
    :param sep: separator of values in source rows
    :param min_freq:
    :return: new columns: most frequent, 'others', 'np.NAN'
    """
    # -- collect all values to one series
    arr = []
    for x in source_series:
        x: str = x
        if x is not np.NAN and sep in x:
            for v in x.split(sep):
                arr.append(v)
        else:
            arr.append(x)
    col = pd.Series(arr)
    # -- find most frequent
    vc_s = pd.value_counts(col)
    vcp_s = ((vc_s / vc_s.sum()) * 100)
    mask = vcp_s[vcp_s < min_freq]  # will be replaced
    mask_saved = vcp_s[vcp_s >= min_freq]
    mask_v = mask.index.values.tolist()
    # -- create template for encoding
    saved_values = mask_saved.index.values.tolist()
    saved_values.append('other')
    saved_values.append(np.NAN)
    full_new_row_pos = {x: i for i, x in enumerate(saved_values)}
    full_new_row_empty = [0 for _ in range(len(saved_values))]
    # -- split each row and replace elements by mask
    retarr = []
    for x in source_series:
        if x is np.NAN or sep not in x:
            v = full_new_row_empty.copy()
            if x in mask_v:
                x = 'other'
            v[full_new_row_pos[x]] = 1
        else:
            v = full_new_row_empty.copy()
            l: list = x.split(sep)
            for y in l:
                if y in mask_v:
                    y = 'other'
                v[full_new_row_pos[y]] += 1
        retarr.append(v)
    # -- replace np.NAN to 'np.NAN'
    saved_values[-1] = 'np.NAN'
    saved_values = [str(source_series.name) + '_' + x for x in saved_values]
    print(saved_values)
    return pd.DataFrame(retarr, columns=saved_values)


def sparse_classes(p, t, id_check, min_categories=60, percent=1):
    """in columns with categories > min_categories replaced categories
    which is < percent with "others" string
    explore: from chepelev_pack.exploring import explore_sparse_classes"""
    df: pd.DataFrame = pd.read_pickle(p)

    cols = df.select_dtypes(include="object").columns.tolist()
    # -- ignore id
    if 'id' in cols:
        cols.remove('id')

    print('sparse_classess process columns:')
    for c in cols:
        vc_s = df[c].value_counts()
        if vc_s.shape[0] > min_categories:
            df[c] = condense_category(df[c], min_freq=percent)
    # print(df.head(10).to_string())
    # exit()

    # -- ids chech
    ids: list = pd.read_pickle(id_check)
    print("ids check:", len(ids), df.shape[0])
    assert all(df['id'] == ids)

    return save(t, df)


def encode_categorical_label(df: pd.DataFrame, columns: list,
                             label_encoders_train: dict=None):
    from sklearn.preprocessing import OrdinalEncoder

    print("LabelEncoder classes:")
    lablel_encoders = {}
    for c in columns:
        # pandas way
        # print(c, dict(enumerate(df[c].astype('category').cat.categories)))
        #
        # sklearn way
        if df[c].isna().sum() > 0:
            print(f"WARNING column {c} has NAN value")
            df[c] = df[c].fillna('')
        if label_encoders_train is None or c not in label_encoders_train.keys():
            # print(c, df[c].unique(a).shape[0])
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  # , dtype=np.float

            df_n = df[c].to_numpy().reshape(-1, 1)
            le.fit(df_n)
            print(c, le.categories_)
            lablel_encoders[c] = le
            df[c] = le.transform(df_n)
        else:  # we ensure the same encoding as in train
            df_n = df[c].to_numpy().reshape(-1, 1)
            df[c] = label_encoders_train[c].transform(df_n)
    return df, lablel_encoders


def encode_categorical_onehot(df, columns: list) -> (pd.DataFrame, list):
    # -- convert type np.nan to str
    for c in columns:

        # sort to have sorted columns (optional)
        aa = df[c].astype(str).unique().sort()
        aa3 = pd.api.types.CategoricalDtype(aa)
        df[c] = df[c].astype(aa3)

    # replace categorical with one-Hot
    df = pd.get_dummies(df, columns=columns, dummy_na=False)  # dummy_na=True - to check for not filled

    # print result
    print("One-Hot result columns:")
    for c in columns:
        print(c, [cc for cc in df.columns if cc.startswith(c)])

    # -- int32/int64 or float to int
    # get new one-hot columns
    new_one_hot_columns = []
    for c in columns:
        c_new = [x for x in df.columns if c in x]
        new_one_hot_columns += c_new

    return df, new_one_hot_columns


def encode_categorical_numberical_twovalues(df, columns: list) -> pd.DataFrame:
    print("Two values with NA columns:")
    for c in columns:
        if df[c].unique().shape[0] == 2 and df[c].isna().sum() > 0:  # if unique: [xxx, NAN]
            not_na = [v for v in df[c].unique().tolist() if not pd.isna(v)][0]
            df[c].fillna(not_na - 1, inplace=True)
            print(c, df[c].unique())
    return df


def encode_categorical(df: pd.DataFrame, one_hot_min=2,
                       onehot_max=10,
                       label_encoders_train: dict = None):
    """
        encode categorical and two-values numerical=(<NA>,x)
        - id field ignored
        column names must be strings!
        :param df:
        :param label_encoders_train:
        :param onehot_max:
        :param one_hot_min:
        :param id_check:
        :param p:
        :return:
        """
    # -- separate strings("object") from numerics
    categorial_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    # print(categorial_columns)
    # print(df.dtypes)
    # exit()
    # -- ignore id
    if 'id' in categorial_columns:
        categorial_columns.remove('id')
    if 'id' in numerical_columns:
        numerical_columns.remove('id')

    # -- separate one-hot and label encoding
    one_hot_columns = []
    label_e_columns = []
    for c in categorial_columns:
        unique_count = df[c].unique().shape[0]
        if one_hot_min < unique_count < onehot_max:  # 10
            one_hot_columns.append(c)  # если будет слишком много значений, то будет слишком много столбцов
        else:
            label_e_columns.append(c)

    # -- encode categorical Ordinary - better for feature importance
    df, label_encoders = encode_categorical_label(df, label_e_columns, label_encoders_train)
    # label_encoders = None

    # -- encode categorical One-Hot
    df, new_one_hot_columns = encode_categorical_onehot(df, one_hot_columns)

    # -- numerical with two columns
    df = encode_categorical_numberical_twovalues(df, numerical_columns)

    print('onehot', new_one_hot_columns)
    print('label', label_e_columns)
    print()

    # convert
    for c in label_e_columns + new_one_hot_columns:
        # print('asdasd', c, df)
        df[c] = df[c].astype(int)
    return df, label_encoders


def encode_categorical_pipe(p_or_df, id_check: str,
                            one_hot_min=2,
                            onehot_max=10,
                            label_encoders_train: dict = None, id_check_f=True) -> (str, list):
    """ column names must be strings! """
    df: pd.DataFrame = load(p_or_df)

    df, label_encoders = encode_categorical(df, one_hot_min=one_hot_min, onehot_max=onehot_max,
                                            label_encoders_train=label_encoders_train)

    df = df.astype(float) #, errors='coerce')

    # -- ids chech
    if id_check_f:
        check_id(df, id_check)
    # -- save ids
    # save('id.pickle', df['id'].tolist())
    # print(df.columns)
    return df, label_encoders


def fill_na(p1: str, t1: str, p2: str = None, t2: str = None, id_check1: str = None, id_check2: str = None) -> (str, str or None):
    """
    1) replace NA for numericla - mean, categorical - most frequent
    2) categorical: nan > 0.6 or unique = 2 - do not fill missing
    3) numerical: unique = 2 and has nan - do not fill missing and save as str
    2) fix types for seaborn - int64 to int, float64 to float

    """
    df1: pd.DataFrame = pd.read_pickle(p1)
    df1_i = df1.index
    if p2:
        df2: pd.DataFrame = pd.read_pickle(p2)
        df2_i = df2.index
        df = pd.concat([df1, df2])
        assert df.shape[0] == (df1.shape[0] + df2.shape[0])
    else:
        df = df1

    # df['CLIENT_WI_EXPERIENCE']
    # return

    # for c in df.columns:
    #     print(c, df[c].isna().sum())
    # from sklearn.impute import SimpleImputer
    # df = SimpleImputer(strategy="most_frequent").transform(df)
    # print(df)
    # return
    # -- separate strings("object") from numerics

    categorial_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_columns = df.select_dtypes(exclude=["object"]).columns.tolist()
    excluded_cols = set()
    # exclude 2 unique columns
    for c in categorial_columns:
        if df[c].unique().shape[0] == 2 or df[c].isna().sum() / df.shape[0] > 0.6:
            excluded_cols.add(c)
    for c in numerical_columns:
        if df[c].unique().shape[0] == 2 and df[c].isna().sum() > 0:  # if unique: [xxx, NAN]
            excluded_cols.add(c)
            # df[c] = df[c].astype('str')

    print("2 unique values columns excluded:", excluded_cols)
    categorial_columns = list(set(categorial_columns) - excluded_cols)
    numerical_columns = list(set(numerical_columns) - excluded_cols)
    # print(categorial_columns)
    # print(numerical_columns)

    # -- ignore id
    if 'id' in categorial_columns:
        categorial_columns.remove('id')
    if 'id' in numerical_columns:
        numerical_columns.remove('id')

    # -- replace na for categorical most frequent or as 'NaN
    print("NA count in categorical columns:")
    [print(c, df[c].isna().sum()) for c in categorial_columns]
    print()
    if categorial_columns:
        print("fill na with mode in categorical:\n", df[categorial_columns].mode().iloc[0])
        print()

    df[categorial_columns] = df[categorial_columns].fillna(df.mode().iloc[0])

    # -- replace na for numerical
    for c in numerical_columns:
        df_nona = df[df[c].notna()][c]
        med = df_nona.median()
        na_count = df[c].isna().sum()
        if na_count:
            print(c, "count:", na_count, "fill na with median:", med)
        if isinstance(med, pd._libs.missing.NAType):
            med = 0
        # print(, type(med), med == np.NAN)
        if str(df[c].dtype).lower().startswith('int'):
            df[c] = df[c].fillna(round(med)).astype(int)
        else:
            print("cast", c)
            df[c] = df[c].fillna(med).astype(float)

    # -- ids chech
    df1 = df.loc[df1_i]
    if p2:
        df2 = df.loc[df2_i]

    if id_check1:
        ids: list = pd.read_pickle(id_check1)
        print("ids check:")
        assert all(df1['id'] == ids)
    if id_check2:
        ids: list = pd.read_pickle(id_check2)
        print("ids check:")
        assert all(df2['id'] == ids)

    # -- save ids
    # save('id.pickle', df['id'].tolist())
    if p2:
        return save(t1, df1), save(t2, df2)
    else:
        return save(t1, df1), None


def standardization(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    # print(df.isna().sum().sum())
    # print(df.dtypes)
    # df.dropna(axis=1, inplace=True)

    # remove save excluded columns
    if exclude is not None:
        df = df.reset_index(drop=True)
        df_exception = df[exclude]
        df = df.drop(exclude, 1)

    # --- first
    # df = (df - df.mean(axis=0)) / df.std(axis=0)
    # --- second
    scale = np.nanstd(df, axis=0)
    df /= scale
    mean = np.nanmean(df, axis=0)
    df -= mean
    # print(df.describe().dtypes)
    # print(df.describe().to_string())
    # return
    # -- thirst way
    # from sklearn import preprocessing
    # df2 = preprocessing.StandardScaler().fit_transform(df)
    # print(df.isna().sum().sum())
    # print(df2.mean(axis=0))
    # df = np.rint(df2)

    # print(pd.DataFrame(df2, dtype=int).dtypes)
    # for i, c in enumerate(exclude):
    #     df[c] = excluded_series[i]
    #     print(c, "was returned")

    if exclude is not None:
        df = pd.concat([df, df_exception], axis=1, sort=False, verify_integrity=True)

    # df = df.fillna(0)
    return df


def standardization_pipe(p, exclude: list = None):
    df: pd.DataFrame = pd.read_pickle(p)
    standardization_pipe(df, exclude=exclude)
    return save('standardized.pickle', df)


def standardization01(p, exclude: list = None):
    df: pd.DataFrame = pd.read_pickle(p)

    # remove save excluded columns
    if exclude is not None:
        df = df.reset_index(drop=True)
        df_exception = df[exclude]
        df = df.drop(exclude, 1)

    n_min, n_max = 0, 1
    minimum, maximum = np.min(df, axis=0), np.max(df, axis=0)
    m = (n_max - n_min) / (maximum - minimum)
    b = n_min - m * minimum

    df = m * df + b

    if exclude is not None:
        df = pd.concat([df, df_exception], axis=1, sort=False, verify_integrity=True)

    return save('standardized.pickle', df)


def remove_single_unique_values(dataframe: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Drop all the columns that only contain one unique value.
    not optimized for categorical features yet.

    """
    cols_to_drop = dataframe.nunique(dropna=False)
    cols_to_drop = cols_to_drop.loc[cols_to_drop.values == 1].index
    cols_to_drop = cols_to_drop.to_list()
    print("single_value dropped", cols_to_drop)
    dataframe.drop(columns=cols_to_drop, inplace=True)
    print("columns after drop:", dataframe.columns.tolist())
    return dataframe, cols_to_drop


def feature_selection(p, na_percent=20):
    """
    remove features with
    1) 1 value
    2) remove > na_percent NA values in category

    :param p:
    :param na_percent: drop columns if NA > na_percent
    :return:
    """

    df: pd.DataFrame = pd.read_pickle(p)
    print("before:", df.shape)
    # print((df.isnull().sum()/df.shape[0]*100).to_string())
    # -- 1 value
    df = remove_single_unique_values(df)
    # -- NAN > 20%
    cs = set(df.columns.tolist())
    perc = na_percent  # Like N %
    min_count = int(((100 - perc) / 100) * df.shape[0] + 1)
    df = df.dropna(axis=1,
                   thresh=min_count)
    cs2 = set(df.columns.tolist())
    cs_diff = cs - cs2
    print("after:", df.shape)
    print("removed by percent:", cs_diff)
    return save('feature_selection.pickle', df)


def feature_selection_cat(p, target:str, na_percent=20):
    """
    remove features with
    1) 1 value
    2) remove > na_percent NA values in category

    :param p:
    :param na_percent: drop columns if NA > na_percent
    :return:
    """
    df: pd.DataFrame = pd.read_pickle(p)
    # [print(c) for c in df.columns if str(c).endswith("IS_CANCEL")]

    # df = df[(df['ander'] == 0) | (df['ander'] == 1)]

    print("before:", df.shape)

    # -- 1 value
    df = remove_single_unique_values(df)
    # -- NAN > 20% in category

    bad_columns = set()
    for cat in df[target].unique():
        df_cat = df[df[target] == cat]
        proc: pd.Series = (df_cat.isna().sum() / df_cat.shape[0]) * 100
        cols = proc[proc > na_percent].index
        bad_columns.update(cols)

    df = df.drop(columns=bad_columns)
    # -- report
    print("removed columns by percent", bad_columns)
    print("after:", df.shape)

    return save('feature_selection.pickle', df)


def feature_selection_cat_rat(p, target: str, non_na_ration_trash=0.3):
    """
    remove features with
    1) 1 value
    2) if na ration for column between categories > non_na_ration_trash we drop that column
    ex. na1 = 10 in cat1, n2 in cat2 = 20 : 10/20 = non_na_ration_trash > 0.3 => drop this column


    :param p:
    :param na_percent: drop columns if NA > na_percent
    :return:
    """
    df: pd.DataFrame = pd.read_pickle(p)
    # [print(c) for c in df.columns if str(c).endswith("IS_CANCEL")]

    df = df[(df['ander'] == 0) | (df['ander'] == 1)]

    print("before:", df.shape)

    # -- 1 value
    df = remove_single_unique_values(df)
    # -- NAN > 20% in category
    # calc
    proc_cats = []
    for cat in df[target].unique():
        df_cat = df[df[target] == cat]  # category slice

        proc_of_not_na: pd.Series = 1 - (df_cat.isna().sum() / df_cat.shape[0])  # * 100  # columns with
        proc_cats.append(proc_of_not_na)

    # compare
    procs = pd.concat(proc_cats, axis=1)
    bad_columns = set()
    bad_columns_zero = set()
    ra = abs(np.log(non_na_ration_trash))
    for c, row in procs.iterrows():
        for i, v1 in enumerate(row):
            for j, v2 in enumerate(row):
                if i == j:
                    continue
                else:
                    if v1 == 0 and v2 == 0:
                        bad_columns_zero.add(c)
                    elif v2 == 0 and abs(np.log(v1)) > ra:
                        bad_columns.add(c)
                    elif v1 == 0 and abs(np.log(v2)) > ra:
                        bad_columns.add(c)
                    elif abs(np.log(v1/v2)) > ra:  # 1/2 = 0.693 # 1/4 = 1.38
                        bad_columns.add(c)

    # -- report
    proc_all: pd.Series = (1- df.isna().sum() / df.shape[0]) * 100
    proc_all.sort_values(inplace=True)

    print("dropped by ration: not na [ not na in categories]")
    r = []
    for c in bad_columns:
        r.append((c, round(proc_all.loc[c],2), np.around(procs.loc[c].to_list(), 2)))
    print(pd.DataFrame(r))

    # -- drop
    df = df.drop(columns=bad_columns_zero)
    df = df.drop(columns=bad_columns)
    print("after:", df.shape)

    return save('feature_selection.pickle', df)


def downsample(X_test, y_test, rate: float = 1, random_seed=1): #-> (np.ndarray, np.ndarray)
    """
    train_rate = sum(y_train) / (len(y_train) - sum(y_train))
    imblearn.over_sampling.RandomOverSampler
    :param X_test: np.ndarray or Dataframe
    :param y_test:
    :param cl: class to downsample
    :param rate: should be greater, rate of other class (if cl==0, other_class=1) default rate=1 class1/class0=1
    :return: X, y
    """
    np.random.seed(random_seed)

    flag_df = False
    if isinstance(X_test, pd.DataFrame):
        flag_df = True

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    import math

    i_cl0 = np.where(y_test == 0)[0]
    i_cl1 = np.where(y_test == 1)[0]
    n_cl0 = len(i_cl0)
    n_cl1 = len(i_cl1)
    current_rate = n_cl1 / n_cl0
    inverted_rate = math.exp(-math.log(rate))
    if rate > current_rate:
        cl = 0
    else:
        cl = 1
    if cl == 0:
        # print(n_cl1, rate, n_cl0)
        # print("RATE", (n_cl1 / (rate * n_cl0)))
        # sel_q = n_cl0 * (n_cl1 / (rate * n_cl0))
        sel_q = (n_cl1 / rate)
        i_class0_downsampled = np.random.choice(i_cl0, size=round(sel_q), replace=False)
        print("remover records from class 0: ", len(i_cl0) - len(i_class0_downsampled))
        if flag_df:
            X = pd.concat([X_test.iloc[i_cl1], X_test.iloc[i_class0_downsampled]],
                          axis=0, ignore_index=True)
        else:
            X = np.vstack((X_test[i_cl1, :], X_test[i_class0_downsampled, :]))
        y = np.hstack((y_test[i_cl1], y_test[i_class0_downsampled]))
    elif cl == 1:
        # print(n_cl1, rate, n_cl0)
        # print("RATE", (n_cl1 / (rate * n_cl0)))
        # sel_q = n_cl1 * (rate * n_cl0 / n_cl1)
        sel_q = rate * n_cl0
        # print("n_cl0", n_cl0)
        # print("sel_q", sel_q)
        # print("i_cl1", sum(i_cl1))
        # select records
        i_class1_downsampled = np.random.choice(i_cl1, size=round(sel_q), replace=False)
        print("remover records from class 1: ", len(i_cl1) - len(i_class1_downsampled))

        if flag_df:
            X = pd.concat([X_test.iloc[i_cl0], X_test.iloc[i_class1_downsampled]],
                             axis=0, ignore_index=True)
        else:
            X = np.vstack((X_test[i_cl0, :], X_test[i_class1_downsampled, :]))
        y = np.hstack((y_test[i_cl0], y_test[i_class1_downsampled]))
    else:
        raise Exception()

    # if flag_df:
    #     X = pd.DataFrame(X, columns=X_columns)
    return X, y


def test_downsample():
    X = np.zeros((100, 30))
    y = np.zeros(100)
    y[:20] = 1  # first 20
    # now 20/100 = 0.2
    # 20 - cl1, 100 - cl0 + cl1
    # -- cl1 higher by selecting from cl0:
    # will be 20/40 = 0.5 inv 2
    # 20*2 = 40 - cl2
    # will be 20/66 = 0.3 inv 3.3
    print(X.shape)
    print(y.shape)
    train_rate = sum(y) / (len(y) - sum(y))
    print("train_rate b ", train_rate)
    X, y = downsample(X, y, rate=0.12)
    train_rate = sum(y) / (len(y) - sum(y))
    print("train_rate a", train_rate)
    # print(X.shape)
    # print(y.shape)
    # print(y.sum())
    assert X.shape == (90, 30)
    assert y.shape == (90,)
    assert y.sum() == 10.0


    # -- cl0 higher by selecting from cl1
    X = np.zeros((100, 30))
    y = np.zeros(100)
    y[:20] = 1  # first 20
    # now 80/100 = 0.8
    # 100 - cl1+cl0, 80 - cl0
    # will be 80/90=0.88 inv 1.13
    train_rate = sum(y) / (len(y) - sum(y))
    print("train_rate", train_rate)
    print(y.sum(), len(y))
    X, y = downsample(X, y, rate=0.88)
    train_rate = sum(y) / (len(y) - sum(y))
    print("train_rate", train_rate)
    # print(X.shape)
    # print(y.shape)
    print(y.sum(), len(y))
    assert X.shape == (43, 30)
    assert y.shape == (43,)
    assert y.sum() == 20.0


def split(p: str, t1: str, t2: str, split_date=None) -> (str, str):
    """
    :param p: source dataframe
    :param t1: train dataframe
    :param t2: test dataframe
    :param split_date:
    :return:
    """
    df: pd.DataFrame = load (p)

    X_train, X_test, y_train, y_test = train_test_split(df, df['ander'], test_size=0.20, shuffle=False,
                                                        random_state=2, stratify=None)

if __name__ == '__main__':
    # test downsample
    test_downsample()
