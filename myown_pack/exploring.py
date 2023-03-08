import pandas as pd
import numpy as np
from chepelev_pack.common import load


def explore_sparse_classes(p, min_categories=60, percent=1):
    """ for columns with categories > min_categories
    show how many records in percent"""
    df: pd.DataFrame = pd.read_pickle(p)
    print(df.dtypes)
    # categorical only
    for c in df.select_dtypes(include="object").columns:
        vc_s = df[c].value_counts()
        print(c, "total features:", vc_s.shape[0])
        if vc_s.shape[0] > min_categories:
            pc = round((df.shape[0] / 100) * percent)

            vcp_s = ((vc_s / vc_s.sum()) * 100)
            mask = vcp_s[vcp_s < percent]
            # values_to_replace = {x: new_name for x in list(mask.index)}
            print("percent", percent, "records:", pc, "categories:", mask.index.shape[0])
            print("features lower percent", vc_s[vc_s < pc].shape[0])
        print((vc_s / vc_s.sum() * 100).head(20).to_string(), '\n')


def explore_na(p):
    """ TODO: use .isna().sum()"""
    df: pd.DataFrame = pd.read_pickle(p)

    # df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    # target = 'ander'
    # df_0 = df[df[target] == 0]  # appr
    # df_1 = df[df[target] == 1]  # rej without 091
    # df_2 = df[df[target] == 2]  # rej with 091

    per: pd.Series = (df.isnull().sum() / df.shape[0] * 100)
    # per_0: pd.Series = (df_0.isnull().sum() / df_0.shape[0] * 100)
    # per_1: pd.Series = (df_1.isnull().sum() / df_1.shape[0] * 100)
    # per_2: pd.Series = (df_2.isnull().sum() / df_2.shape[0] * 100)

    per.name = 'Процент от Всех'
    # per_0.name = 'В Одобренных'
    # per_1.name = 'В отклоненных андером'
    # per_2.name = 'В отклоненных клиентом'

    per = per.round(2)
    # per_0 = per_0.round(2)
    # per_1 = per_1.round(2)
    # per_2 = per_2.round(2)
    # per = pd.concat([per, per_0, per_1, per_2], axis=1)
    # per = per[per['Процент от Всех'] > 0]
    per.sort_values() #(by=['Процент от Всех'], axis=0, inplace=True, ascending=False)
    print("Процент NA от всех записей")
    print(per.to_string())
    per.to_csv("Процент пустых значений.csv")

    return



    # df['CAR_INFO_IS_EPTS'].isna().sum()
    # return
    # print(per)
    # print(df['PAY_FOR_CAR_RICIPIENT'].unique())
    # return
    # print(per.tail(1).index[0])
    # print(per)
    # return

    df.fillna(-94, inplace=True)  # groupby required

    target='ander'
    df_0 = df[df[target] == 0]  # appr
    df_1 = df[df[target] == 1]  # rej without 091
    df_2 = df[df[target] == 2]  # rej with 091


    # na_appr: pd.Series = (df['ander'])

    # c = 'CAR_INFO_NUMBER_PTS'
    # print(df_0['CLIENT_WI_EXPERIENCE'].isna().sum())

    cou = []
    for c in per[per > 0].index:
        # df_c_0 = df_0[c].isnull().sum()
        # if df_0
        # if c == 'CLIENT_WI_EXPERIENCE':
        #     print(df_0.groupby(c).size ()[-94])
        try:
            p_0 = round(df_0.groupby(c).size()[-94] / df_0.shape[0] * 100, 2)
        except ZeroDivisionError:
            p_0 = 99.99999
        except (KeyError, IndexError):
            p_0 = 0
        try:
            p_1 = round(df_1.groupby(c).size()[-94] / df_1.shape[0] * 100, 2)
        except ZeroDivisionError:
            p_1 = 99.99999
        except (KeyError, IndexError):
            p_1 = 0
        try:
            p_2 = round(df_2.groupby(c).size()[-94] / df_2.shape[0] * 100, 2)
        except (KeyError, IndexError):
            p_2 = 0
        # df_c['Отношение одоб/откл'] = round(np.log(df_c['count_0'] / df_c['count_1']), 5)
        try:
            o = round(np.log(p_0 / p_1), 5)
        except ZeroDivisionError:
            o = np.NAN
        # print(c, o, p_0, p_1, p_2)
        cou.append((c, o, p_0, p_1, p_2))
    df = pd.DataFrame(data=cou, columns=['c', 'Отношение одоб/откл', 'процент_Одоб', 'процент_Откл', 'процент_ОтклП'])

    df = df.set_index('c')
    df = df.join(per.round(2))
    df = df.sort_values(by='Отношение одоб/откл')
    df = df.sort_values(by='Отношение одоб/откл')
    df.to_csv("na_explore.csv")
    print(df.to_string())
    exit(0)
    # import numpy as np
    # print(df.groupby(c).size().index.unique())
    # df_c_0 = df.groupby(c).size() #.loc[np.nan]
    # print(df_c_0.index.unique())
    # # df_c_0 = df.groupby(c).size()[df_c_0.index.isnull()]
    # print(df_c_0)
    # print(type(df[c].iloc[1002]))

    # df = df.join()
    df_res = []
    # print(per.index)
    # v = []
    # for c, p in per.items():
    #     if c in fields_dict:
    #         v.append((c, round(p, 1), fields_dict[c]))
    #     else:
    #         v.append((c, round(p, 1), ''))
    #         print(c)
    #     print(c, '\t', str(round(p, 1)) + '%')
    # pd.DataFrame(v).to_csv('miss deal_status_release.csv')


def explore_columns_and_na(p, fields_dict):
    from chepelev_pack.plot import rename_columns
    df: pd.DataFrame = pd.read_pickle(p)
    # print(df.columns)
    # return

    proc_all: pd.Series = np.around((df.isna().sum() / df.shape[0]) * 100, 2)

    df_proc_all = pd.DataFrame(proc_all) #.reset_index(drop=False)
    df_proc_all.rename(columns={0: 'Процент пропущенных'}, inplace=True)
    #proc_all.sort_values(inplace=True)
    # print(proc_all.to_string())
    # renamed columns
    for k in df_proc_all.index:
        if k not in fields_dict:
            fields_dict[k] = k

    new_col:pd.Series = df_proc_all.index.map(fields_dict)
    # print(new_col)

    c = pd.concat([df_proc_all.reset_index(), new_col.to_frame().reset_index(drop=True)], axis=1)
    c.to_csv('columns_selected.csv')
    print(c.to_string())


def corr_analysis(p, target:str = None, method="pearson"):
    """

    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'dejavu'
    # rcParams['font.sans-serif'] = ['DejavuSans']
    plt.rc('font', size=6)

    df: pd.DataFrame = pd.read_pickle(p)

    # from chepelev_pack.plot import histogram_two_in_one
    # histogram_two_in_one(df.tail(500), 'MBKI_SCORING' , 'МБКИ_исп_пр')
    # a = df['MBKI_SCORING'].tolist()
    # print(np.std(a))
    # a = df['MBKI_SCORING'].to_numpy()/2
    # print(np.std(a))
    # exit(0)

    # -- Pearson for continuous features


    corr = df.corr(method=method)
    if target:
        print("correlation to target")
        print(corr[target].sort_values().to_string())
        print()
    # mask # clear diagonal
    mask = np.zeros_like(corr, int)
    np.fill_diagonal(mask, 1)
    corr.iloc[mask.astype(bool)] = 0

    # -- max correlation plot
    corr_max = np.nanmax(corr.to_numpy().astype(float), axis=1)
    print(corr_max)
    d = pd.DataFrame({'c':corr_max, 'a':df.columns.to_numpy()}).sort_values(by='c', ascending=False)

    sns.barplot(x='c',y='a',data=d, orient='h')
    plt.title("max correlation")
    plt.show()
    # -- full correlation plot
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(15, 15))
        ax = sns.heatmap(corr, square=True, linewidths=.8, cmap="YlGnBu")  # mask=mask,
        ax.set_title("Главная диаграмма корреляции")
    # plt.savefig('a.png')
    # -- (> 0.98) for 1 and ( 0.2 > and < 0.98) for 2
    # filter low correlation
    lim_max = 0.98
    lim_min = 0.2
    df_too_much = None  # (> 0.98) for 1
    df_h = None  # ( 0.2 > and < 0.98) for 2
    for c in corr.columns:
        print(c, abs(corr[c].max()), abs(corr[c].min()))

        if lim_min < abs(corr[c].max()) or lim_min < abs(corr[c].min()):
            if df_h is None:
                df_h = pd.DataFrame(df[c])
            else:
                df_h[c] = df[c]
            # df.drop(c, 1, inplace=True)
        if abs(corr[c].max()) > lim_max or abs(corr[c].min()) > lim_max:
            if df_too_much is None:
                df_too_much = pd.DataFrame(df[c])
            else:
                df_too_much[c] = df[c]
            # df.drop(c, 1, inplace=True)

    # mask for cleared corr
    corr = df_h.corr(method=method)
    # mask # clear diagonal
    mask = np.zeros_like(corr, int)
    np.fill_diagonal(mask, 1)
    corr.iloc[mask.astype(bool)] = 0

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(15, 15))
        ax = sns.heatmap(corr, square=True, linewidths=.8, cmap="YlGnBu")  # mask=mask,
        ax.set_title("Имевшие > 0.2 корреляцию на главной диаграмме")
    # plt.show()
    # -- plot (> 0.98) for 1
    if df_too_much is not None:
        corr = df_too_much.corr(method=method)
        # mask # clear diagonal
        mask = np.zeros_like(corr, int)
        np.fill_diagonal(mask, 1)
        corr.iloc[mask.astype(bool)] = 0

        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(15, 15))
            ax = sns.heatmap(corr, square=True, linewidths=.8, cmap="YlGnBu")  # , mask=mask
            ax.set_title("Значения имевшие > 0.98 корреляцию на главной диаграмме")
            f.subplots_adjust(left=0.49, bottom=0.4)
    plt.show()
    # Из пар Столбцов которые коррелируют между собой с большим значением, мы удалим один из них, который дает меньший
    # вклад с точки зрения важности параметров для линейной модели.


def frequency_analysis(p, target='ander'):
    """ for binary target"""
    import matplotlib.pyplot as plt
    df = load(p)
    df.drop(columns=['id'], inplace=True)
    print("Всего:", df.shape[0])
    df_0 = df[df[target] == 0]
    df_1 = df[df[target] == 1]
    df_0.drop(columns=['ander'], inplace=True)
    df_1.drop(columns=['ander'], inplace=True)
    print("0", df_0.shape[0])
    print("1", df_1.shape[0])
    print("NA 0:\n", df_0.isna().sum().to_string())
    print("NA 1:\n", df_1.isna().sum().to_string())

    # print(df['MBKI_SCORING'].value_counts().to_string())
    # exit()

    # -- unique < 40
    min40_dfs = []
    # -- hist
    other_c = []
    for i, c in enumerate(df_0.columns):
        if df[c].unique().shape[0] > 40:
            other_c.append(c)
            continue
        df_c_0 = df_0.groupby(c).size().reset_index(
            name="0")
        df_c_1 = df_1.groupby(c).size().reset_index(
            name="1")
        df_c_0['count'] = df_c_0["0"] / df_0.shape[0]
        df_c_1['count'] = df_c_1["1"] / df_1.shape[0]
        # print(df_c_0)  # значения и частота принятые # approved
        # print(df_c_1)  # значения и частота отвергнутые
        df_c_0.set_index(c, verify_integrity=True, inplace=True)
        df_c_1.set_index(c, verify_integrity=True, inplace=True)
        df_c = df_c_0.join(df_c_1, lsuffix='_0', rsuffix='_1')
        # df_c['0/1'] = df_c.apply(f, axis=1)
        df_c['1/0'] = round(np.log(df_c['count_1']/df_c['count_0']), 5)
        df_c.drop(columns=['count_0', 'count_1'], inplace=True)
        min40_dfs.append(df_c)
    # -- print
    [print(df_c.to_markdown()) for df_c in min40_dfs]
    # -- to CSV
    # for df_c in min40_dfs:
    #     with open('a.csv', 'a') as f:
    #         df_c.to_csv(f)
    #         f.write('\n')
    # -- bar plot
    for df_c in min40_dfs:
        # rename title
        # if df_c.index.name in fields_dict:
        #     df_c.index.name = fields_dict[df_c.index.name]
        # df_c.plot.bar(rot=10)
        df_c.plot.bar(rot=10)
        plt.ylabel("кол-во")
        plt.tight_layout()
        # plt.show()
        plt.savefig('bar_norm_' + df_c.index.name)
    # return
    # -- unique > 40
    print("columns unique > 40:", other_c)
    for c in other_c:
        f, ax = plt.subplots(figsize=(6, 4))
        df_0_c = df_0[c]
        df_1_c = df_1[c]
        # plt.hist(x=df_1, bins=10, color='red', alpha=0.6, normed=True)
        df_0_c = pd.to_numeric(df_0_c, errors='coerce')
        df_1_c = pd.to_numeric(df_1_c, errors='coerce')
        print(df_0_c.tail(3).to_string())
        df_0_c.hist(ax=ax, bins=20, color='red', alpha=0.6, density=True, label='0')  # , stacked=True
        df_1_c.hist(ax=ax, bins=20, color='green', alpha=0.6, density=True, label='1')  # , stacked=True
        plt.legend()
        plt.title(c, {'fontsize': 20})
        # plt.show()
        plt.savefig('hist_norm ' + c)
        plt.close()
    exit(0)


def shap(p):
    import shap
    import shap.maskers
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    from sklearn.utils import resample
    import matplotlib.ticker as plticker
    # shap.initjs()
    # -- data
    df: pd.DataFrame = pd.read_pickle(p)
    # -- special cases
    # df = df[df['МБКИ_треб_пассп'] != 1]
    # df = df[df['МБКИ_недост'] != 1]
    df = df[df['МБКИ_данные_не'] != 1]
    # df = df[df['МБКИ_розыск'] != 1]
    # df = df[df['МБКИ_налспецуч'] != 1]
    #  --RENAME COLUMNS:


    target = 'ander'
    X = df.drop([target, 'id'], 1)

    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # ,         shuffle=False, stratify=False, random_state=0)
    # -- selected rows
    X_rows = shap.utils.sample(X_train, 1000)  # random
    # X_rows:  = shap.kmeans(X_test, 100) # for KernelExplainer
    print(df.columns)

    # m = RandomForestClassifier(class_weight={0: 2, 1: 1}, max_depth=9, n_estimators=330)
    # Precision: 0.554395  # Recall: 0.319218
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, gamma=1, gpu_id=-1, eval_metric='logloss',
    #                   importance_type='gain', interaction_constraints='',
    #                   learning_rate=0.02, max_delta_step=0, max_depth=6,
    #                   min_child_weight=1, missing=np.nan, monotone_constraints='()',
    #                   n_estimators=80, n_jobs=2, nthread=2, num_parallel_tree=1,
    #                   random_state=42, reg_alpha=0.5, reg_lambda=1,
    #                   scale_pos_weight=0.75, seed=42, subsample=1, # 1.3346062351320898
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
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
    # Precision: 0.698925 # Recall: 0.016287
    # m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #                   colsample_bynode=1, colsample_bytree=0.5, eval_metric='logloss',
    #                   gamma=2, gpu_id=0, importance_type='gain',
    #                   interaction_constraints='', learning_rate=0.300000012,
    #                   max_delta_step=0, max_depth=3, min_child_weight=6, missing=np.nan,
    #                   monotone_constraints='()', n_estimators=100, n_jobs=2, nthread=2,
    #                   num_parallel_tree=1, random_state=22, reg_alpha=0.2, reg_lambda=1,
    #                   scale_pos_weight=0.34263295553618134, seed=22, subsample=1,
    #                   tree_method='exact', use_label_encoder=False,
    #                   validate_parameters=1, verbosity=1)
    print("train model")
    m.fit(X_train, y_train)

    ex = shap.TreeExplainer(m)  # data=shap.maskers.Independent(X, max_samples=1000)


    # -- interaction values all
    # interaction_values = ex.shap_interaction_values(X_rows, y, tree_limit=5)
    # shap.summary_plot(interaction_values, X_rows, max_display=7, plot_size=(13, 13), show=True)  # , plot_type="dot"
    # # plt.savefig("shap_interaction_values_xgboost.png") # require resize
    # plt.show()
    # return

    print(" -- calc shap values", X_rows.shape)
    shap_values = ex.shap_values(X_rows, check_additivity=True, approximate=True)

    # -- plot bees features importance
    shap.summary_plot(shap_values, X_rows, plot_size=(10, 8))  # plot_type="bar"
    # plt.savefig("shap_values_xgboost.png") # require left
    # -- dependence scatter plot for 2 features
    # shap_values = ex(X_rows, check_additivity=True)
    # shap.plots.scatter(shap_values[:, 'AUTO_DEAL_INITIAL_FEE'], color=shap_values[:, "EQUIFAX_SCORING"], hist=True)
    # plt.show()
    # return


    # -- force for one recrod
    # shap_values = ex(X_rows, check_additivity=True)
    # plt.switch_backend('agg')
    # shap.force_plot(ex.expected_value, shap_values.values[2, :], X.iloc[2, :], matplotlib=True, show=False))
    # plt.savefig("a.jpg")
    # return

    # -- dependence plot and single feature plot
    # for c in X.columns:
    #     # shap.(c, shap_values, X_test, show=False)
    #     shap.dependence_plot(c, shap_values, X_rows, show=False)
    #     plt.savefig('shap_xgboost_'+c+'.png')
    # return

    # -- functions
    def interaction_values_two(pair: tuple):
        rows = resample(X_test, n_samples=1000, random_state=2)
        interaction_values = ex.shap_interaction_values(rows, tree_limit=5)  # , tree_limit=5
        shap.dependence_plot(pair, interaction_values, rows, show=False)
        plt.savefig('shap_inter_'+pair[0]+'_' + pair[1]+'.png')

    def shapv_plot_two_features_raw():
        c1 = 'OKB_RATING_SCORING_КИ отсутствует'  # binary
        c2 = 'EQUIFAX_SCORING'
        v20 = resample(X[X[c1] == 0], n_samples=1000)
        v21 = resample(X[X[c1] == 1], n_samples=1000)
        shap_values0 = ex.shap_values(v20, check_additivity=True, approximate=True)
        shap_values1 = ex.shap_values(v21, check_additivity=True, approximate=True)
        c2p = df.columns.tolist().index(c2)
        int_v0 = shap_values0[:, c2p]  # ==0
        int_v1 = shap_values1[:, c2p]  # ==1
        fig, ax = plt.subplots(figsize=(10, 8))
        import matplotlib.ticker as plticker
        loc = plticker.MultipleLocator(base=50)
        ax.xaxis.set_major_locator(loc)
        ax.scatter(v21[c2], int_v1, color='r', alpha=0.3, s=4, label='OKB_RATING_SCORING_КИ отсутствует=1')
        ax.scatter(v20[c2], int_v0, color='g', alpha=0.3, s=4, label='OKB_RATING_SCORING_КИ отсутствует=0')
        ax.grid()
        plt.legend()
        plt.xlabel(c2)
        plt.ylabel('Важность')
        plt.title(c1 + ' в взаимодействии с ' + c2)
        plt.savefig(c1 + ' в взаимодействии с ' + c2)

    def shapv_plot_two_features(shap_values, pair):
        " pair = (v1,v2) or pair = (v1,v1)"
        print("shapv_plot_two_features", pair)
        # from time import sleep
        if type(pair[1]) != str:
            for x in pair[1]:
                shapv_plot_two_features(shap_values, (pair[0], x))
                # sleep(2)
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        # loc = plticker.MultipleLocator(base=50)
        # ax.xaxis.set_major_locator(loc)
        ax.grid()
        # ax.set_ylabel("Вес")
        # ax.set_alpha(0.3)
        # , interaction_index=None - no color
        if pair[0] == pair[1]:
            shap.dependence_plot(pair[0], shap_values, X_rows, ax=ax, show=False, interaction_index=None)
        else:
            shap.dependence_plot(pair[0], shap_values, X_rows, ax=ax, show=False,
                                 interaction_index=pair[1])
        plt.savefig('shap_xgboost_' + pair[0] + '&' + pair[1] + '.png')
        plt.close()

    # ------------ CALL --------------
    # pair = ('EQUIFAX_SCORING', 'OKB_RATING_SCORING_КИ отсутствует')
    # interaction_values_two(pair)

    # pair = ('CLIENT_WI_EXPERIENCE', ['OKB_RATING_SCORING_КИ отсутствует', 'EQUIFAX_SCORING'])
    # pair = ('EQUIFAX_SCORING', ['AUTO_DEAL_INITIAL_FEE', 'CLIENT_WI_EXPERIENCE'])
    # pair = ('AUTO_DEAL_INITIAL_FEE', ['OKB_RATING_SCORING_КИ отсутствует', 'EQUIFAX_SCORING', 'CLIENT_WI_EXPERIENCE',
    #                                   'CLIENT_MARITAL_STATUS'])
    # pair = ('OKB_RATING_SCORING_КИ отсутствует', 'OKB_RATING_SCORING_КИ отсутствует')
    # pair = ('Первоначальный взнос', 'Первоначальный взнос')
    # pair = ('Скоринг Бюро Эквифакс 4Score', 'Скоринг Бюро Эквифакс 4Score')
    # pair = ('Скоринговый балл ОКБ', 'Скоринговый балл ОКБ')
    # pair = ('Стаж в организации (мес.)', 'Стаж в организации (мес.)')
    # pair = ('Семейное положение', 'Семейное положение')
    # pair = ('Скоринговый балл клиента (анкетный скоринг)', 'Скоринговый балл клиента (анкетный скоринг)')
    pair = ('Скоринговый балл НБКИ', ('Скоринговый балл НБКИ', 'Скоринг Бюро Эквифакс 4Score', 'OKB_RATING_SCORING_КИ отсутствует', 'OKB_RATING_SCORING_Хорошая КИ'))
    # pair = ('МБКИ_требаналотч', ('Скоринговый балл НБКИ', 'Скоринг Бюро Эквифакс 4Score', 'OKB_RATING_SCORING_КИ отсутствует',
    # 'OKB_RATING_SCORING_Хорошая КИ'))
    # pair = ('МБКИ_треб_исп_пр', ('Скоринговый балл НБКИ', 'Скоринг Бюро Эквифакс 4Score', 'OKB_RATING_SCORING_КИ отсутствует',
    #                      'OKB_RATING_SCORING_Хорошая КИ'))
    # pair = ('МБКИ_нет_огр', ('Скоринговый балл НБКИ', 'Скоринг Бюро Эквифакс 4Score', 'OKB_RATING_SCORING_КИ отсутствует',
    #                     'OKB_RATING_SCORING_Хорошая КИ'))
    # pair = ('МБКИ_треб_адрес', ('Скоринговый балл НБКИ', 'Скоринг Бюро Эквифакс 4Score', 'OKB_RATING_SCORING_КИ отсутствует',
    #                      'OKB_RATING_SCORING_Хорошая КИ'))
    # pair = ('INITIAL_FEE_DIV_CAR_PRICE','INITIAL_FEE_DIV_CAR_PRICE')
    pair = ('AUTO_DEAL_INITIAL_FEE', 'AUTO_DEAL_INITIAL_FEE')
    shapv_plot_two_features(shap_values, pair)
    plt.show()
    return

    # -- two with first binary interaction
    def not_shap_hist_two_with_one_binary():
        pair = ('OKB_RATING_SCORING_КИ отсутствует', 'EQUIFAX_SCORING')
        c = pair[1]
        fig, ax = plt.subplots(figsize=(10,3))
        import matplotlib.ticker as plticker
        loc = plticker.MultipleLocator(base=50)
        ax.xaxis.set_major_locator(loc)
        df = df[df['OKB_RATING_SCORING_КИ отсутствует'] == 1]
        df_1 = df[df['ander'] == 1][c]
        df_0 = df[df['ander'] == 0][c]
        df_1.hist(ax=ax, bins=30, color='red', alpha=0.6, density=True, label='отклоненные')  # , stacked=True
        df_0.hist(ax=ax, bins=30, color='green', alpha=0.6, density=True, label='акцептованные')  # , stacked=True
        plt.legend()
        plt.title('OKB_RATING_SCORING_КИ отсутствует=1')
        plt.savefig('OKB_RATING_SCORING_КИ отсутствует=1' + str(pair))
    # return
    # TODO
    # shap.plots.force(ex.expected_value, shap_values)
    # plt.show()

    def hist_ander():
        feture_importance = (
            'OKB_RATING_SCORING_КИ отсутствует',
            'EQUIFAX_SCORING',
            'CLIENT_WI_EXPERIENCE',
            'OKB_RATING_SCORING_Хорошая',
            'NBKI_SCORING',
            'AUTO_DEAL_INITIAL_FEE',
            'OKB_SCORING',
            'NBKI_RATING_SCORING_КИ',
            'ANKETA_SCORING',
            'CLIENT_MARITAL_STATUS',
            'МБКИ_тапоМБКИ',
            'NBKI_RATING_SCORING_Хорошая',
            'MBKI_SCORING',
            'CLIENT_AGE',
            'МБКИ_неогр',
            'МБКИ_исп_пр',
            'EQUIFAX_RATING_SCORING_КИ',
            'EQUIFAX_RATING_SCORING_Хорошая',
            'CLIENT_DEPENDENTS_COUNT',
            'МБКИ_адрес',
            'МБКИ_недост',
            'OKB_RATING_SCORING_Нейтральная',
            'NBKI_RATING_SCORING_Нейтральная',
            'EQUIFAX_RATING_SCORING_Нейтральная',
            'МБКИ_невыполнена',
            'EQUIFAX_RATING_SCORING_Ошибка',
            'МБКИ_спецуч',
            'NBKI_RATING_SCORING_Ошибка',
            'OKB_RATING_SCORING_Ошибка',
            'МБКИ_розыск',
            'OKB_RATING_SCORING_Плохая',
            'NBKI_RATING_SCORING_Плохая',
            'МБКИ_суд',
            'EQUIFAX_RATING_SCORING_Плохая'
        )
        for c in other_c:
            ax = plt.gca()
            df_1 = df[df['ander'] == 1][c]
            df_0 = df[df['ander'] == 0][c]
            # plt.hist(x=df_1, bins=10, color='red', alpha=0.6, normed=True)
            df_1.hist(ax=ax, bins=20, color='red', alpha=0.6, density=True, label='отклоненные')  # , stacked=True
            df_0.hist(ax=ax, bins=20, color='green', alpha=0.6, density=True, label='акцептованные')  # , stacked=True
            plt.legend()
            plt.title(c)
            # plt.show()
            plt.savefig('hist_norm ' + c)
            plt.close()


def compare_identical_dataframes(df1, df2, id_field: str = 'id'):
    """
    calc changes in every row

    :param df1:
    :param df2:
    :param id_field:
    :return:
    """
    print("df1.columns", df1.columns.tolist())
    print('shapes df1, df2', df1.shape[0], df2.shape[0])
    print("columns difference:")
    diff = []
    # check index unique
    assert df1[id_field].unique().shape[0] == df1.shape[0]
    assert df2[id_field].unique().shape[0] == df2.shape[0]
    # sort and generate new index
    df1 = df1.sort_values(by=[id_field]).reset_index(drop=True)
    df2 = df2.sort_values(by=[id_field]).reset_index(drop=True)
    assert (df1[id_field] != df2[id_field]).sum() == 0
    # compare columns types
    # for c in df1.columns:
    #     diff.append((c,(df1[c] != df2[c]).sum()))
    # -- calc number of rows with changes per column
    for c in df1.columns:
        diff.append((c, df1[c].compare(df2[c]).shape[0]))
        # print(c, df1[c].compare(df2[c]).shape[0])
    for v in sorted(diff, key=lambda x: x[1], reverse=True):
        print(v)


def compare_different_dataframes(df1, df2):
    """compare by describe and calc mean for every column
    p1,p2:   df: pd.DataFrame = pd.read_pickle(p)
    """
    from chepelev_pack.common import encode_categorical
    from chepelev_pack.common import standardization
    df1, label_encoders = encode_categorical(df1)  # 1 or 0 # fill_na required

    df2, label_encoders = encode_categorical(df2,
                                             label_encoders_train=label_encoders)  # 1 or 0 # fill_na required
    df1 = standardization(df1)
    df2 = standardization(df2)

    df1 = (df1 - df1.mean()) / df1.std()
    df2 = (df2 - df2.mean()) / df2.std()
    des1 = df1.describe()
    des2 = df2.describe()
    diff = []
    # print(des1['DC_REJECTED_CODES_DES'])
    assert des1.columns.shape == df1.columns.shape
    assert des2.columns.shape == df2.columns.shape
    cols_inter = set(des1.columns.tolist()).intersection(set(des2.columns.tolist()))
    for c in cols_inter:
        col_diff = []
        for r in ['mean', 'std', '25%', '50%', '75%']:  # des1.index.tolist()
            # save diff for specific column
            print(des1[c])
            print(des1[c].loc[r])
            col_diff.append(abs(des1[c].loc[r] - des2[c].loc[r]))
        mean_difference = np.mean(np.array(col_diff))
        # save to sort later
        diff.append((c, mean_difference))
    # diff = np.array(diff)
    # mask = np.isnan(diff[:,1].astype(float))
    # ma = np.ma.masked_array(diff[:,1], mask=mask)
    # diff = diff[np.argsort(ma)]
    diff = pd.DataFrame(diff, columns=['1','2']) # .fillna(-1)
    # print(sorted(diff, key=lambda x: x[1], reverse=True))
    # print(diff.columns.tolist())
    print(diff.sort_values(by=['2'], ascending=False).to_string())
    # for v in diff.sort_values(by=['2']).iterrows():
    #     print(v)


if __name__ == '__main__':
    # df1 = pd.DataFrame({'col1':np.array([1,2,3,4,5]), 'col2':np.array([4,np.NAN,6,7,8]), 'col3':[4,5,5,7,8]})
    # df2 = pd.DataFrame({'col1': np.array([1, 2, 3, 4, 5]), 'col2': np.array([4, np.NAN, 6, 7, 8]), 'col3': [4, 5, 6, 7, 8]})
    # compare_dataframes(df1, df2, id_field='col1')
    # df1 = load('/home/u2/h4/PycharmProjects/evseeva/slice_20_2110.pickle')
    # df2 = load('/home/u2/h4/PycharmProjects/evseeva/slice4_20_2110.pickle')
    df = load('/home/u2/h4/PycharmProjects/evseeva/mart_norma_auto6.csv')
    df = df[(df['DEAL_CREATED_DATE'] > '2020-01-01')]
    df = df[(df['DEAL_CREATED_DATE'] < '2022-02-10')]  # year , month, date

    df1 = df[(df['DEAL_CREATED_DATE'] > '2021-08-21')]
    df1 = df1[(df1['DEAL_CREATED_DATE'] < '2021-12-10')]  # year , month, date

    df2 = df[(df['DEAL_CREATED_DATE'] > '2021-12-10')]
    df2 = df2[(df2['DEAL_CREATED_DATE'] < '2022-01-10')]  # year , month, date

    # compare_dataframes(df1, df2, id_field='APP_SRC_REF')
    compare_different_dataframes(df1, df2)

