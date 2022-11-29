from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# own


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


def impute(df_o: pd.DataFrame) -> pd.DataFrame:
    """
    All numeric columns myst be int32 or int64 or int
    1) Separate categorical, numerical and bad_numerical columns
    2) Encode categorical columns (dummy a,b - 0,1)
    3) Concat encoded and numerical columns
    3) predict missing values (imputation)
    4) decode categorical columns
    """
    # -- object ot str type
    # categorial_columns2 = df_o.select_dtypes(include=["object"]).columns
    # for c in categorial_columns2:
    #     df_o[c] = df_o[c].astype(str)
    # print(df_o.isna().sum())
    # return
    # COLUMNS WITH COUNT(NaN) > 50%
    exception_columns = []
    for c in df_o:
        if df_o[c].hasnans:
            if df_o[c].isna().sum() > (df_o.shape[0]*0.7):  # NaN > 70%
                exception_columns.append(c)
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
        elif c in exception_columns:
            numerical_columns_excluded.append(c)
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

    # CONCAN NUMERICAL AND ENCODED CATECOGORICAL
    df = pd.concat([n_df, c_df], axis=1, sort=False, verify_integrity=True)

    # IMPUTATION
    # from sklearn.ensemble import ExtraTreesRegressor
    # ExtraTreesRegressor()
    imp = IterativeImputer(random_state=0, verbose=2,
                           max_iter=2, n_nearest_features=None, sample_posterior=True)
    # TEMPORAL REPLACEMENT
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # df = df.to_numpy()
    for c in numerical_columns:
        if n_df[c].dtype.name == 'object':
            print(c, n_df[c].dtype.name)  # required
            raise Exception

    df: np.array = imp.fit_transform(df)
    # RESTORE NUMERICAL COLUMN TYPES and NAMES
    df = pd.DataFrame(df)  # numpy to pandas

    df.columns = numerical_columns + categorical_columns  # restore column names
    for c in df:  # columns
        if c in numerical_columns:
            df[c] = df[c].round().astype(columns_dtypes[c])  # float or int
        else:  # categorical
            alph = label_encoders[c].transform(label_encoders[c].classes_)  # original set without NaN
            if len(alph) != 0:
                df[c] = pd.Series(closest(alph, df[c])).astype(int)  # replace not original to closest original
            else:
                print("wtf", alph)
    # print("c5", df['autocredit_car_info.`condition`'].tail(10).to_string())
    # Add numerical columns with NaN > 50%
    df = df.join(n_e_df)
    # df.reset_index(drop=True, inplace=True)
    # DECODE CATEGORICAL COLUMNS
    for c in categorical_columns:
        df[c] = label_encoders[c].inverse_transform(df[c])

    return df


def impute2(df_o: pd.DataFrame) -> pd.DataFrame:
    """
    All numeric columns myst be int32 or int64 or int
    1) Separate categorical, numerical and bad_numerical columns
    2) Encode categorical columns (dummy a,b - 0,1)
    3) Concat encoded and numerical columns
    3) predict missing values (imputation)
    4) decode categorical columns
    """
    from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer
    # si = SingleImputer()  # pass through data once
    mi = MultipleImputer()  # pass through data multiple times
    # mice = MiceImputer()  # pass through data multiple times and iteratively optimize imputations in each column

    # -- object ot str type
    # categorial_columns2 = df_o.select_dtypes(include=["object"]).columns
    # for c in categorial_columns2:
    #     df_o[c] = df_o[c].astype(str)
    # print(df_o.isna().sum())
    # return
    # COLUMNS WITH COUNT(NaN) > 50%
    exception_columns = []
    for c in df_o:
        if df_o[c].hasnans:
            if df_o[c].isna().sum() > (df_o.shape[0]*0.7):  # NaN > 70%
                exception_columns.append(c)
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
        elif c in exception_columns:
            numerical_columns_excluded.append(c)
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

    # CONCAN NUMERICAL AND ENCODED CATECOGORICAL
    df = pd.concat([n_df, c_df], axis=1, sort=False, verify_integrity=True)

    # IMPUTATION
    # from sklearn.ensemble import ExtraTreesRegressor
    # ExtraTreesRegressor()
    # imp = IterativeImputer(random_state=0, verbose=2,
    #                        max_iter=6, n_nearest_features=None, sample_posterior=True)
    imp = mi
    # TEMPORAL REPLACEMENT
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # df = df.to_numpy()
    for c in numerical_columns:
        if n_df[c].dtype.name == 'object':
            print(c, n_df[c].dtype.name)  # required
            raise Exception

    df: np.array = imp.fit_transform(df)
    # RESTORE NUMERICAL COLUMN TYPES and NAMES
    df = pd.DataFrame(df)  # numpy to pandas

    df.columns = numerical_columns + categorical_columns  # restore column names
    for c in df:  # columns
        if c in numerical_columns:
            df[c] = df[c].round().astype(columns_dtypes[c])  # float or int
        else:  # categorical
            alph = label_encoders[c].transform(label_encoders[c].classes_)  # original set without NaN
            if len(alph) != 0:
                df[c] = pd.Series(closest(alph, df[c])).astype(int)  # replace not original to closest original
            else:
                print("wtf", alph)
    # print("c5", df['autocredit_car_info.`condition`'].tail(10).to_string())
    # Add numerical columns with NaN > 50%
    df = df.join(n_e_df)
    # df.reset_index(drop=True, inplace=True)
    # DECODE CATEGORICAL COLUMNS
    for c in categorical_columns:
        df[c] = label_encoders[c].inverse_transform(df[c])

    return df


if __name__ == '__main__':
    # LOAD
    df_o: pd.DataFrame = pd.read_pickle('features.pickle')
    df_o = df_o.iloc[3000:8000]
    # print(df_o['autocredit_car_info.`condition`'].tail(10).to_string())
    print(df_o.tail(10).to_string())

    # df = impute(df_o)
    # # print(df['autocredit_car_info.`condition`'].tail(10).to_string())
    # print(df.tail(10).to_string())
    # print(df.shape)
    #
    # # SAVE
    # pd.to_pickle(df, "imputed_features.pickle")
