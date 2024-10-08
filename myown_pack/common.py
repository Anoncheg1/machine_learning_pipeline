import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import NearestNeighbors

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


def save(sp: str, obj:any, format = 'pandas') -> str:
    """
    save to .pickle
    :param sp: must be *.pickle
    :param obj: sp
    :param format: pandas, numpy
    :return:
    """
    if format == 'pandas':
        pd.to_pickle(obj, sp) # .pkl
    elif format == 'numpy':
        # .npy allow_pickle=False - for security
        np.save(sp, obj, allow_pickle=False, fix_imports=False)
    if type(obj) == np.ndarray:
        args = (sp, obj.shape)
    elif type(obj) == pd.DataFrame:
        args = (sp, obj.shape, obj.columns.tolist())
    else:
        args = (sp,)
    print('-- save --', *args)
    print()
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
    if 'id' in df.columns:
        assert df['id'].tolist() == ids
    else:
        assert df.index.tolist() == ids


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
    " require id field "
    print("-- OUTLIERS_NUMERICAL")
    # from scipy.stats import skew
    df: pd.DataFrame = load(p)
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
        df.drop(df_filtered.index, inplace=True) # remove filtered rows from main df

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
    df.reset_index(drop=True, inplace=True)
    save('id_train.pickle', df.index.tolist())
    # save('id_train.pickle', df['id'].tolist())

    print("filtered:", df_deleted)
    print("total filtered count:", tc)
    return save('without_outliers.pickle', df)


def values_byfreq(col: pd.Series or pd.DataFrame, min_freq=0.4):
    """ sparse classes lower min_freq
    return two lists: >= min_freq and < min_freq
    used in conjunction with sklearn.preprocessing.OneHotEncoder """
    # vc_s = pd.value_counts(col) # old
    assert isinstance(col, pd.Series)
    vc_s = col.value_counts()
    vcp_s = vc_s / vc_s.sum() # unique value with percent in 0.xx
    print("vcp_s", vcp_s)
    # if isinstance(col, pd.Series):
    return [x for x in vcp_s[vcp_s >= min_freq].index.tolist()], [x for x in vcp_s[vcp_s < min_freq].index.tolist()]
    # elif isinstance(col, pd.DataFrame):
    #     return [x[0] for x in vcp_s[vcp_s >= min_freq].index.tolist()], [x[0] for x in vcp_s[vcp_s < min_freq].index.tolist()]


def condense_category_byfreq(col: pd.Series, min_freq=0.4, new_name='other'):
    """ sparse classes lower min_freq percent replaced with new_name
    multicolumn: from chepelev_pack.common import sparse_classes"""
    # vc_s = pd.value_counts(col) # old
    vc_s = col.value_counts()
    vcp_s = vc_s / vc_s.sum() # unique value with percent in 0.xx
    print("vcp_s", vcp_s)
    mask: pd.Series = vcp_s[vcp_s < min_freq]
    print(col.name, 'count of low frequent values:', mask.count(), '\t\tcount of saved values:', vcp_s[vcp_s >= min_freq].count())
    values_to_replace = {x: new_name for x in list(mask.index)}
    return col.replace(values_to_replace)

def condense_category_bycount(col: pd.Series, max_category_count=1, new_name='other'):
    """ sparse classes lower min_freq percent replaced with new_name
    multicolumn: from chepelev_pack.common import sparse_classes"""
    # vc_s = pd.value_counts(col) # old
    vc_s = col.value_counts()
    vcp_s = ((vc_s / vc_s.sum()) * 100)
    # top = vcp_s[:max_category_count]
    bottom = vcp_s[max_category_count:]
    # print(sel_good)
    # mask: pd.Series = vcp_s[vcp_s < min_freq]
    # print(col.name, 'count of low frequent values:', mask.count(), '\t\tcount of saved values:', vcp_s[vcp_s >= min_freq].count())
    values_to_replace = {x: new_name for x in list(bottom.index)}
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


def sparse_classes(p, t=None, id_check=None, min_categories=60, percent=1, cols=None):
    """in columns with categories > min_categories replaced categories
    which is < percent with "others" string
    explore: from chepelev_pack.exploring import explore_sparse_classes"""
    df: pd.DataFrame = load(p)
    if cols is None:
        cols = df.select_dtypes(include="object").columns.tolist()
        if 'id' in cols: # ignore id
            cols.remove('id')



    print('sparse_classess process columns:')
    for c in cols:
        vc_s = df[c].value_counts()
        if vc_s.shape[0] > min_categories:
            df.loc[:, c] = condense_category_byfreq(df[c], min_freq=percent)
    # print(df.head(10).to_string())

    # -- ids chech
    if id_check:
        check_id(df, id_check)
    if t:
        return save(t, df)
    else:
        return df

def encode_categorical_label_old(df: pd.DataFrame, columns: list,
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


def encode_categorical_label(df: pd.DataFrame,
                             columns: list,
                             encoders: dict or OrdinalEncoder=None,
                             min_frequency=0.5):
    print("LabelEncoder:")
    dfn = df.copy()
    report_new_columns = columns
    # - ensure there is one encoder for all columns
    if encoders is not None:
        print("columns", columns)
        print("encoders", encoders)
        if isinstance(encoders, dict):
            encoder = encoders[columns[0]]
            for k, v in encoders.items():
                if k in columns:
                    assert v == encoder
        elif isinstance(encoders, OrdinalEncoder):
            encoder = encoders
        else:
            logging.critical("encoders is not list and not OrdinalEncoder.")
        assert isinstance(encoder, OrdinalEncoder)

    else: # encoder is None: # - get encoder
        encoder: OrdinalEncoder = OrdinalEncoder(
            min_frequency=min_frequency) # all that have < min_frequency will be as 'others' column
        encoder.fit(dfn[columns])
    print("infrequent_categories",
          {k:v for k, v in zip(columns, encoder.infrequent_categories_)})
    print()
    # - create net columns
    new_columns = encoder.categories_[0].tolist()
    for x in encoder.infrequent_categories_[0].tolist():
        new_columns.remove(x)

    # - transform
    print(columns, encoder)
    transformed = encoder.transform(dfn[columns])
    # - Create a Pandas DataFrame of the hot encoded column
    ohe_df = pd.DataFrame(transformed, columns=columns)
    # - concat with original data
    dfn = pd.concat([dfn.drop(columns, axis=1), ohe_df], axis=1)

    print(dfn.head(3).to_string())
    # # print result
    print("One-Hot result columns:")
    print(report_new_columns)
    # -- int32/int64 or float to int
    new_columns = report_new_columns
    return dfn, encoder, new_columns

def encode_categorical_onehot_old(df, columns: list, encoders: dir) -> (pd.DataFrame, list):
    # # -- convert type np.nan to str
    for c in columns:
        # sort to have sorted columns (optional)
        aa = df[c].astype(str).unique().sort()
        aa3 = pd.api.types.CategoricalDtype(aa)
        df[c] = df[c].astype(aa3)

    # replace categorical with one-Hot (old)
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


def encode_categorical_onehot(df, columns: list, encoders: dir=None, min_frequency=0.2) -> (pd.DataFrame, list):
    """ encoders may be mix of encoders.
    we encode every column with different encoder
    return (new_df, encoders:dict[column, encoder], new_columns)"""
    print("encode_categorical_onehot:")
    dfn = df.copy()
    new_encoders = {}
    report_new_columns = {}
    # - encode:
    for c in columns:
        # - get encoder or create and train
        if encoders is None or c not in encoders:
            encoder: OneHotEncoder = OneHotEncoder(
                handle_unknown='infrequent_if_exist', # nan as infrequent 'other'
                sparse_output=False,
                min_frequency=min_frequency) # all that have < min_frequency will be as 'others' column
            encoder.fit(dfn[c].to_numpy().reshape(-1, 1))
            # print("infrequent categories for:", c, len(encoder.infrequent_categories_))
            # print()
            new_encoders[c] = encoder
        else:
            encoder = encoders[c]
        # - create net columns
        print("encoder.categories_.shape", len(encoder.categories_[0]))
        new_columns = encoder.categories_[0].tolist()
        if encoder.infrequent_categories_[0] is not None:
            new_columns.append('other')
            for x in encoder.infrequent_categories_[0].tolist():
                if x in new_columns:
                    new_columns.remove(x)
        new_columns = [c + '_' + x for x in new_columns]
        report_new_columns[c] = new_columns
        # - transform
        transformed = encoder.transform(dfn[c].to_numpy().reshape(-1, 1))
        # - Create a Pandas DataFrame of the hot encoded column
        # print("c", c)
        # print("transformed.shape", transformed.shape)
        # print("len(new_columns)", len(new_columns), new_columns)
        # print("drop_idx_", encoder.drop_idx_)
        # print("encoder.infrequent_categories_[0]", encoder.infrequent_categories_[0])
        ohe_df = pd.DataFrame(transformed, columns=new_columns)
        # - concat with original data
        dfn = pd.concat([dfn, ohe_df], axis=1).drop([c], axis=1)
    # print(dfn.head(3).to_string())
    # print result
    print("One-Hot result columns:")
    # for c in columns:
    #     print(c, [cc for cc in dfn.columns if cc.startswith(c)])
    for k, v in report_new_columns.items():
        print(k,v)
    # -- int32/int64 or float to int
    new_columns = []
    for v in report_new_columns.values():
        for x in v:
            new_columns.append(x)
    if len(new_encoders) == 0:
        return dfn, encoders, new_columns
    else:
        return dfn, new_encoders, new_columns


def encode_categorical_numerical_twovalues(df, columns: list) -> pd.DataFrame:
    print("Two values with NA columns:")
    for c in columns:
        if df[c].unique().shape[0] == 2 and df[c].isna().sum() > 0:  # if unique: [xxx, NAN]
            not_na = [v for v in df[c].unique().tolist() if not pd.isna(v)][0]
            df[c].fillna(not_na - 1, inplace=True)
            print(c, df[c].unique())
    print()
    return df


def encode_categorical(df: pd.DataFrame,
                       encoders: dict = None,
                       min_frequency=0.3):
    """Encode categorical and two-values numerical=(<NA>,x)
        - id field ignored
        column names must be strings!

    steps for first run:
    1.
    steps for second-test run:
    1.
        :param df or path:
        :param encoders: {'column name': OneHotEncoder() or other, ...}
        :param min_frequency
        :return:
        """
    # -- test if NaN exist
    if df.isna().sum().sum() > 0:
        logging.error("There is NaN values in DataFrame:", df.isna().sum())


    one_hot_columns = []
    label_e_columns = []
    # -- separate strings("object") from numerics
    categorial_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    # -- ignore id
    if 'id' in categorial_columns:
        categorial_columns.remove('id')
    if 'id' in numerical_columns:
        numerical_columns.remove('id')

    if encoders is None: # first time for training
        # -- separate one-hot and label encoding
        for c in categorial_columns:
            unique_count = df[c].unique().shape[0]
            if 2 < unique_count: # do not ecode two values as one hot
                one_hot_columns.append(c)  # если будет слишком много значений, то будет слишком много столбцов
            else:
                label_e_columns.append(c)
        # -- one_hot columns that have 3 unique values after min_frequency
        one_hot_bad = []
        for c in one_hot_columns:
            f, r = values_byfreq(df[c], min_freq=min_frequency)
            if len(f) + 1 <= 3:
                one_hot_bad.append(c)
        [one_hot_columns.remove(c) for c in one_hot_bad]
        [label_e_columns.append(c) for c in one_hot_bad]


    else: # second time for test
        # -- get names of columns
        for k, v in encoders.items():
            if isinstance(v, OrdinalEncoder):
                label_e_columns.append(k)
            elif isinstance(v, OneHotEncoder):
                one_hot_columns.append(k)
            else:
                logging.critical("Unknown encoder!")
        for c in numerical_columns[:]:
            if c in label_e_columns or c in one_hot_columns:
                numerical_columns.remove(c)
    print('label columns', label_e_columns)
    print('onehot columns', one_hot_columns)
    print('numerical columns', numerical_columns)
    print()
    # -- encode categorical Ordinary - better for feature importance
    new_l_columns = []
    label_encoders = {}
    if len(label_e_columns) > 0:
        # - encode
        x = encode_categorical_label(df, label_e_columns,
                                     encoders, # None or OrdinalEncoder
                                     min_frequency=min_frequency)
        df, label_encoder, new_l_columns = x
        # - create dictionary column -> encoder
        print("after l encoder:", label_e_columns, new_l_columns)
        print()
        label_encoders = {c:label_encoder for c in new_l_columns} # one encoder for all

    # -- encode categorical One-Hot
    new_onehot_columns = []
    onehot_encoders = {}
    if len(one_hot_columns) > 0:
        x = encode_categorical_onehot(df,
                                      one_hot_columns, # required
                                      encoders, # may be mix of encoders
                                      min_frequency=min_frequency)
        df, onehot_encoders, new_onehot_columns = x
        print("onehot_encoders", onehot_encoders)
    # -- numerical with two columns
    if len(numerical_columns) > 0:
        df = encode_categorical_numerical_twovalues(df, numerical_columns)

    print('label', new_l_columns)
    print('onehot', new_onehot_columns)
    print()

    # - convert to int
    for c in new_l_columns + new_onehot_columns:
        df[c] = df[c].astype(int)
    print("before encoders", onehot_encoders, label_encoders)
    onehot_encoders.update(label_encoders)
    print("final encoders", onehot_encoders)
    return df, onehot_encoders


def encode_categorical_pipe_old(p_or_df,
                            one_hot_min=2,
                            onehot_max=10,
                            label_encoders_train: dict = None,
                            id_check: str=None,
                            p_save:str=None, min_frequency = 0.3) -> (str, list):
    """ column names must be strings!
    :p_save minimum frequency below which a category will be considered infrequent."""
    df: pd.DataFrame = load(p_or_df)

    df, label_encoders = encode_categorical_old(df, one_hot_min=one_hot_min,
                                            onehot_max=onehot_max,
                                            label_encoders_train=label_encoders_train,
                                            min_frequency=min_frequency)

    df = df.astype(float) #, errors='coerce')

    # -- ids chech
    if id_check:
        check_id(df, id_check)
    # -- save ids
    # save('id.pickle', df['id'].tolist())
    # print(df.columns)
    if p_save:
        return save(p_save, df), label_encoders
    else:
        return df, label_encoders


def encode_categorical_pipe(p_or_df,
                            encoders_train: dict = None,
                            id_check: str=None,
                            p_save:str=None, min_frequency = 0.3) -> (str, list):
    """ column names must be strings!
    :p_save minimum frequency below which a category will be considered infrequent."""
    print("-- ENCODE_CATEGORICAL_PIPE")
    df: pd.DataFrame = load(p_or_df)

    df, encoders = encode_categorical(df,
                                            encoders=encoders_train,
                                            min_frequency=min_frequency)

    df = df.astype(float) #, errors='coerce')

    # -- ids chech
    if id_check:
        check_id(df, id_check)
    # -- save ids
    # save('id.pickle', df['id'].tolist())
    # print(df.columns)
    if p_save:
        return save(p_save, df), encoders
    else:
        return df, encoders


def fill_na(p1, t1: str, p2 = None, t2: str = None, id_check1: str = None, id_check2: str = None) -> (str, str or None):
    """
    1) replace NA for numericla - mean, categorical - most frequent
    2) categorical: nan > 0.6 or unique = 2 - do not fill missing
    3) numerical: unique = 2 and has nan - do not fill missing and save as str
    2) fix types for seaborn - int64 to int, float64 to float

    Note: Mode for categorical will be bad if column has not many uniques.

    """
    df1: pd.DataFrame = load(p1)
    df1_i = df1.index
    if p2:
        df2: pd.DataFrame = load(p2)
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

    # old:
    df[categorial_columns] = df[categorial_columns].fillna(df.mode().iloc[0])
    # new: TODO
    # from sklearn.impute import KNNImputer
    # nan = np.nan
    # df[categorial_columns]
    # # X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    # imputer = KNNImputer(n_neighbors=2, weights="uniform")
    # imputer.fit_transform(X)

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
        print("ids check:", len(ids), df.shape[0])
        assert df1.index.tolist() == ids
    if id_check2:
        ids: list = pd.read_pickle(id_check2)
        print("ids check:", len(ids), df.shape[0])
        assert df2.index.tolist() == ids

    if p2:
        return save(t1, df1), save(t2, df2)
    else:
        return save(t1, df1)


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


def split(p: str, t1: str, t2: str, target_col=None, date_col=None, perc=0.2, split_date=None) -> (str, str):
    """ saves indexes to id_train.pickle and id_test.pickle
    :param p: source dataframe
    :param t1: train dataframe
    :param t2: test dataframe
    :param split_date:
    :return:
    """
    print("-- SPLIT")
    df: pd.DataFrame = load (p)
    # -- test ids
    try:
        ids: list = pd.read_pickle('id.pickle')
        print("ids check:", len(ids), df.shape[0])
        if 'id' in df.columns:
            assert all(df['id'].tolist() == ids)
        else:
            assert all(df.index.tolist() == ids)
    except FileNotFoundError:
        logging.warn("id.pickle was not found.")
    # -- split
    if len(df[target_col].value_counts()) > 20: # if too many classes or it is regression
        X_train, X_test, y_train, y_test = train_test_split(df, df[target_col], test_size=perc, shuffle=True,
                                                        random_state=2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(df, df[target_col], test_size=perc, shuffle=True,
                                                        random_state=2, stratify=df[target_col])

    df_train = X_train
    df_test = X_test

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    save('id_train.pickle', df_train.index.tolist())
    save('id_test.pickle', df_test.index.tolist())

    return save(t1, df_train), save(t2, df_test)


def SMOTE(T, N:int, k:int):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """
    n_minority_samples, n_features = T.shape # rows, columns

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    NN = N//100
    print(N/100, n_minority_samples)
    n_synthetic_samples = round(NN * n_minority_samples) # 20%
    print(n_synthetic_samples, n_features)
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    print("S.shape", S.shape)

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    print("n_minority_samples", n_minority_samples) # i - 0-> rows
    print("N", N) # n - 0 -> N
    # - for each source row
    for i in range(n_minority_samples): # per row in source
        # get most same rows
        nn = neigh.kneighbors([T[i]], return_distance=False)
        # - repeat for how many we need
        for n in range(NN): # 2
            # - what row we will copy
            # nn_index = nn[0][k-n-1]
            nn_index = nn[0][np.random.randint(1, k-1)]
            #NOTE: nn includes T[i], we don't want to select it
            # c = k-1
            # while nn_index == i:
            #     # nn_index = choice(nn[0])
            # - new row will be between this and same one.
            dif = T[nn_index] - T[i] # row
            gap = np.random.random()
            # [i,:] - row
            S[i*NN + n, :] = T[i,:] + gap * dif[:]
            # S[n + i, :] = T.iloc[i].to_numpy() + gap * dif[:]
            # -i -n1
            #    -n2
            # -i -n1 2+1
            #    -n2
    return S

def split_datetime(df:pd.DataFrame, column:str):
    " replace column with dayofweek, hour, month, quarter"
    # - correct type
    df[column] = pd.to_datetime(df[column])
    # - feature engineering
    new_column = column[(len(column)//2):]
    df[new_column + '_dfw'] = df[column].dt.dayofweek
    df[new_column + '_hour'] = df[column].dt.hour
    df[new_column + '_month'] = df[column].dt.month
    df[new_column + '_quarter'] = df[column].dt.quarter
    df[new_column + '_dofy'] = df[column].dt.dayofyear
    df[new_column + '_monthall'] = df[column].dt.year//2000 + 1/df[column].dt.month
    # - remove columns
    return df.drop(columns=[column])



if __name__ == '__main__':
    # test downsample
    test_downsample()
