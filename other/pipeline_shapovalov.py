import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# own
from chepelev_pack.common import *
from mart_procs import *
from chepelev_pack.model_analysis import drop_collinear_columns
from chepelev_pack.model_analysis import forest_search_parameters
from chepelev_pack.model_analysis import xgb_search_parameters
from chepelev_pack.model_analysis import search_parameters_own_model
from chepelev_pack.model_analysis import drop_collinear_columns

target = 'TT'

if __name__ == '__main__':

    df = pd.read_csv('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto8.csv')
    df2 = pd.read_csv('/home/u2/Desktop/a.csv')
    df2['TT'].fillna(0,inplace=True)
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
    df['FIRST_DECISION_STATE'] = LabelEncoder().fit_transform(df['FIRST_DECISION_STATE'])
    # for c in df.columns.tolist():
    #     df[c] = df[c].astype(int)
    print("wtf3")
    print(df.dtypes.to_string())
    # print(df['TT'].unique())
    # print(df['TT'])
    # exit()
    # exit()
    # df = df.drop(columns=['APP_SRC_REF', '2.949', '2.951', '2.948', '2.944'])
    df = df.drop(columns=['APP_SRC_REF'])
    # df = df[[target, '2.932']]
    # print(df.head(19).to_string())
    # print(df.dtypes.to_string())
    # exit()
    # print(df[df['TARGET']==1].shape)
    # print(df['TARGET'].isna().sum())
    # df = df.fillna(0)
    # exit()
    # print(df.columns)
    df['id'] = df.index
    # -- cat and num
    # print(df.dtypes)
    # print(df.tail(10).to_string())
    # exit()
    p = 'id_train.pickle'
    p = save(p, df['id'])

    # print(df.columns.tolist())
    p = 'shapovalov.pickle'
    p = save(p, df)

    p = outliers_numerical(p, 0.0006,
                            ignore_columns=[], target=target)  # require fill_na for skew test
    p = 'without_outliers.pickle'

    p = fill_na(p1=p, t1='fill_na.pickle', id_check1='id_train.pickle')[0]
    p = 'fill_na.pickle'

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
        'OPTIONAL_KASKO_CANCEL_DATE',
        'DEAL_CONTRACT_NUMBER',
        'DEAL_STATUS',
        'DEAL_CREATED_AT',
    ]

    df: pd.DataFrame = pd.read_pickle(p)
    df.drop(columns=del_cols, inplace=True)
    p = 'shapovalov.pickle'
    p = save(p, df)


    p = sparse_classes(p, 'sparse_classes.pickle', id_check='id_train.pickle')
    p = 'sparse_classes.pickle'
    # df: pd.DataFrame = pd.read_pickle(p)
    # print(df.dtypes.to_string())

    df, label_encoders = encode_categorical_pipe(p, id_check='id_train.pickle', one_hot_min=2,
                                                  onehot_max=10, id_check_f=True)  # 1 or 0 # fill_na required
    # exit()
    # [print(x) for x in df.columns.to_list()]
    # print(df)
    # exit()
    # print(df.columns.tolist())

    # df = pd.read_pickle(p)
    df = remove_single_unique_values(df)[0]
    # print(df.isna().sum().to_string())
    # print(df.shape)
    # exit()

    # cols = df.columns.tolist()
    # cols.remove('score_bank')  # 'SCORE_BANK_NN'
    # cols.remove('SCORE_BANK_NN')
    # df = df[cols]
    print(df.dtypes.to_string())
    df['TT'] = df['TT'].astype(int)
    print(df['TT'].unique())
    # exit()
    # df.drop(columns=['1.18', '2.16', '2.42'], inplace=True)
    # df.drop(columns=['FIRST_DECISION_STATE'], inplace=True)
    p = 'without_col.pickle'
    save(p, df)
    from mart_procs import feature_importance_composite_permut
    from mart_procs import feature_importance_composite_plot

    # p = feature_importance_composite_permut(p, target, nrep_forst=3, nrep_xgboost=3)
    p = 'feature_importance_comp.pickle'
    feature_importance_composite_plot(p)
    # from chepelev_pack.exploring import corr_analysis
    # corr_analysis(p, target=target, method="pearson")
    # # df.drop(columns=['1.14', '1.18', 'score_bank', '2.42', '2.17', '2.16'])
    # # exit()
    #
    # # p2 = drop_collinear_columns(p, target=target, lim_max=0.94, drop=[], )  # and 1 or 0 ander method='spearman'
    #
    # p = feature_importance_composite(p, target)  # outliers_numerical feature_selection fill_na sparse_classes encode_categorical
    # p = 'feature_importance_comp.pickle'
    # feature_importance_composite_plot(p)
