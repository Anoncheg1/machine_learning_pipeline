import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# own
from myown_pack.common import impute_v, outliers_numerical, encode_categorical, save


def hierarchical_clustering_post_anal(labels: list, p):
    df: pd.DataFrame = pd.read_pickle(p)
    print("clasters count " + str(len(set(labels))))
    assert df.shape[0] == len(labels)
    df['labels'] = labels
    dfs = []
    for i in set(labels):
        print(i, list(labels).count(i))
        dfs.append(df[df['labels'] == i])
    for i in set(labels):
        print(i)
        print(dfs[i].describe().to_string())
    megafon = list()
    pdo = list()
    for i in set(labels):
        megafon.append(dfs[i]['Мегафон'])
        pdo.append(dfs[i]['Подтвержденный доход клиента'])
        # df_megafon['Мегафон' + str(i)] = dfs[i]['Мегафон']
        # print(dfs[i]['Мегафон'])

        # df['Мегафон']
        plt.boxplot(megafon)
    plt.title('Мегафон')
    plt.show()
    plt.boxplot(pdo)
    plt.title('Подтвержденный доход клиента')
    plt.show()
    # print(df_megafon.describe())
    # dfs[i].boxplot(column=['Мегафон'])
    # from matplotlib import pyplot as plt
    # plt.show()
    # print(df_1.to_string())
    # print(df_2.describe().to_string())
    #


def plot_scatter_ekv_summa(p):
    df: pd.DataFrame = pd.read_pickle(p)

    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    s_field = ['Запрошенная сумма кредита', 'Эквифакс 4Score']
    labels = ['акцептованные', 'отклоненные', '091 и 01 код отказа', 'выпушенные', ] # 'все заявки'
    colors = ['blue', 'red', 'orange', 'green', ] # 'black'

    groups = [df[df['first_decision_state'] == 0],
              df[df['first_decision_state'] == 1],
              df[(df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)],
              df[df['Статус заявки'] == 'Заявка выпущена'],
              df
              ]

    fig, ax = plt.subplots()
    for i, l in enumerate(labels):
        al = 0.5
        if l == 'отклоненные':
            al = 0.2
        plt.scatter(groups[i]['Запрошенная сумма кредита'].fillna(0),
                    groups[i]['Эквифакс 4Score'].fillna(0), c=colors[i], label=l, alpha=al, s=5)

    fig.set_figwidth(17)
    fig.set_figheight(10)
    plt.legend()
    plt.show()
    plt.xlabel('Запрошенная сумма кредита')
    plt.ylabel('Эквифакс 4Score')
    plt.savefig('Эквифакс к Сумме кредита Диаграмма рассеивания')
    plt.close()


def plot_ekv_summa_2dgist_and_two_gists(p):
    df: pd.DataFrame = pd.read_pickle(p)

    # NA = 0
    df[['Запрошенная сумма кредита', 'Эквифакс 4Score']] = df[['Запрошенная сумма кредита', 'Эквифакс 4Score']].fillna(0)
    # remove 0
    df = df[df['Запрошенная сумма кредита'] != 0]
    df = df[df['Эквифакс 4Score'] != 0]

    x = df['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df['Эквифакс 4Score'].to_numpy().flatten()

    print(x)
    print(y)

    from hist_scatter import scatter_hist2d, scatter_hist_mat
    fig, ax = plt.subplots()

    lims = [0, 3000000]
    bins1 = np.linspace(lims[0], lims[1], 200)
    lims = [0, 1000]
    bins2 = np.linspace(lims[0], lims[1], 200)
    scat = scatter_hist2d(x, y, bins=[bins1, bins2], ax=ax, s=40)
    plt.colorbar(scat)

    fig.set_figwidth(17)
    fig.set_figheight(10)
    plt.xlabel('Запрошенная сумма кредита')
    plt.ylabel('Эквифакс 4Score')
    # plt.show()
    plt.savefig('Эквифакс к Сумме кредита Диаграмма 2д гистограмма')
    plt.close()

    # -- with 2 gistograms
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    plt.xlabel('Запрошенная сумма кредита')
    plt.ylabel('Эквифакс 4Score')
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    scatter_hist_mat(x, y, ax, ax_histx, ax_histy)
    plt.savefig('Эквифакс к Сумме кредита две гистограммы')


def _plot_2dgist_and_two_gists_asone(x, y, title, x_l, y_l, cmap):

    from hist_scatter import scatter_hist2d, scatter_hist_mat

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.03, hspace=0.03)

    ax = fig.add_subplot(gs[1, 0])

    lims1 = [np.min(np.abs(x)), np.max(np.abs(x))]
    bins1 = np.linspace(lims1[0], lims1[1], 40)
    lims2 = [np.min(np.abs(y)), np.max(np.abs(y))]
    bins2 = np.linspace(lims2[0], lims2[1], 40)
    print('x', bins1[5]-bins1[4])
    print('y', bins2[5] - bins2[4])
    # scat = scatter_hist2d(x, y, bins=(bins1, bins2), ax=ax, s=20, cmap=cmap)
    h = ax.hist2d(x, y, bins=(bins1, bins2), cmap=cmap)  # plt.cm.Blues # , norm=colors.LogNorm()

    # ax.set_xlim(left=lims1[0], right=lims1[1])
    # ax.set_xlim(left=500000, right=2600000)
    # ax.set_ylim(bottom=500)
    # ax.set_facecolor('magenta')  # "#  # , facecolor=

    # -- two gists
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # bins1 = np.linspace(lims1[0], lims1[1], 150)
    # bins2 = np.linspace(lims2[0], lims2[1], 50)
    ax_histx.hist(x, bins=bins1)

    ax_histy.hist(y, bins=bins2, orientation='horizontal')
    # plt.colorbar(scat)
    plt.colorbar(h[3])
    ax.set_xlabel(x_l)
    ax.set_ylabel(y_l)

    plt.title(title)
    plt.savefig(title)
    # plt.show()
    plt.close()


def plot_ekv_summa(p):
    from grid_data_plot import griddata_plot
    df: pd.DataFrame = pd.read_pickle(p)
    # NA = 0
    df[['Запрошенная сумма кредита', 'Эквифакс 4Score']] = df[['Запрошенная сумма кредита', 'Эквифакс 4Score']].fillna(
        0)
    # remove 0
    # df = df[(df['Запрошенная сумма кредита'] > 500000) & (df['Запрошенная сумма кредита'] < 2600000)]
    # df = df[df['Эквифакс 4Score'] > 500]
    df = df[df['Запрошенная сумма кредита'] != 0]
    df = df[df['Эквифакс 4Score'] != 0]

    # -- всего
    title = 'com Эквифакс к Сумме кредита '
    x_l = 'Запрошенная сумма кредита'
    y_l = 'Эквифакс 4Score'

    df_appr = df[df['first_decision_state'] == 0]  # 0 - 'approved'
    df_rej = df[df['first_decision_state'] == 1]  # 'rejected'
    df_user_r01091 = df[(df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)]
    # df_user_r01 = df[df['Коды отказа'] == '01']
    # df_user_r091 = df[df['Коды отказа'].str.startswith('091', na=False)]
    df_vip = df[df['Статус заявки'] == 5]  # 'Заявка выпущена'

    # ax.bar(x, df_appr, width=width, color='blue')
    # ax.bar(x + width, df_user_r01091, width=width, color='orange')
    # ax.bar(x + width * 2, df_vip, width=width, color='green')
    # ax2.bar(x + width * 3, df_ob, width=width, color='black')
    # ax2.bar(x + width * 4, df_rej, width=width, color='red')

    x = df['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df['Эквифакс 4Score'].to_numpy().flatten()
    griddata_plot(x, y, 'griddata ' + title + 'всего', x_l, y_l, plt.cm.binary)
    _plot_2dgist_and_two_gists_asone(x, y, title + 'всего', x_l, y_l, plt.cm.binary)

    x = df_appr['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df_appr['Эквифакс 4Score'].to_numpy().flatten()
    griddata_plot(x, y, 'griddata ' + title + 'одобрено', x_l, y_l, plt.cm.Blues)
    _plot_2dgist_and_two_gists_asone(x, y, title + 'одобрено', x_l, y_l, plt.cm.Blues)
    x = df_rej['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df_rej['Эквифакс 4Score'].to_numpy().flatten()
    griddata_plot(x, y, 'griddata ' + title + 'отклонено', x_l, y_l, plt.cm.Reds)
    _plot_2dgist_and_two_gists_asone(x, y, title + 'отклонено', x_l, y_l, plt.cm.Reds)
    x = df_user_r01091['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df_user_r01091['Эквифакс 4Score'].to_numpy().flatten()
    griddata_plot(x, y, 'griddata ' + title + 'отказ клиента 091 и 01', x_l, y_l, plt.cm.Oranges)
    _plot_2dgist_and_two_gists_asone(x, y, title + 'отказ клиента 091 и 01', x_l, y_l, plt.cm.Oranges)
    x = df_vip['Запрошенная сумма кредита'].astype(int).to_numpy().flatten()
    y = df_vip['Эквифакс 4Score'].to_numpy().flatten()
    griddata_plot(x, y, 'griddata ' + title + 'выпущенные', x_l, y_l, plt.cm.Greens)
    _plot_2dgist_and_two_gists_asone(x, y, title + 'выпущенные', x_l, y_l, plt.cm.Greens)


def a(x: str):
    if pd.isna(x):
        return x
    else:
        if x.startswith('Idle timeout') \
                or x.startswith('Couldn\'t') \
                or x.startswith('Server error') \
                or x.startswith('ERROR_NOT_FOUND') \
                or x.startswith('Was not possible') \
                or x.startswith('ERROR_') \
                or x.startswith('Сетевой уровень') \
                or x.startswith('cURL error') \
                or x.startswith('<html>') \
                or x.startswith('Проверка на Uapi'):
            return 'Error'
        else:
            return x


def plot_ki_scoring(p):
    from grid_data_plot import griddata_plot
    df: pd.DataFrame = pd.read_pickle(p)
    print(df['МБКИ'].unique())

    df['МБКИ'] = df['МБКИ'].apply(a)  # clear errors
    import re

    df['МБКИ_адрес'] = df['МБКИ'].apply(lambda x:
                                    re.search('Требуется проверка адреса регистрации/фактического', x) is not None
                                    if pd.notna(x) else False)
    df['МБКИ_исп_пр'] = df['МБКИ'].apply(lambda x:
                                        re.search('Требуется проверка данных о наличии исполнительных про', x) is not None
                                        if pd.notna(x) else False)
    df['МБКИ_неогр'] = df['МБКИ'].apply(lambda x:
                                         re.search('Нет ограни',
                                                   x) is not None
                                         if pd.notna(x) else False)
    df['МБКИ_недост'] = df['МБКИ'].apply(lambda x:
                                        re.search('Недостаточно данных для выполнения пр', x) is not None or
                                        re.search('Данные по клиенту не найд', x) is not None
                                        if pd.notna(x) else False)
    df['МБКИ_розыск'] = df['МБКИ'].apply(lambda x:
                                         re.search('Лица, находящиеся в р',
                                                   x) is not None
                                         if pd.notna(x) else False)
    df['МБКИ_невыполнена'] = df['МБКИ'].apply(lambda x:
                                         re.search('Проверка не выполнена',
                                                   x) is not None
                                         if pd.notna(x) else False)
    df['МБКИ_спецуч'] = df['МБКИ'].apply(lambda x:
                                              re.search('Наличие информации о постановке клиента на спец',
                                                        x) is not None
                                              if pd.notna(x) else False)
    df['МБКИ_паспорт'] = df['МБКИ'].apply(lambda x:
                                         re.search('Требуется проверка информации о действительности па',
                                                   x) is not None
                                         if pd.notna(x) else False)
    print(df['МБКИ'].unique())
    print(df['МБКИ_адрес'].unique())
    return

    df_appr = df[df['first_decision_state'] == 0]  # 0 - 'approved'
    df_rej = df[df['first_decision_state'] == 1]  # 'rejected'
    df_user_r01091 = df[(df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)]
    # df_user_r01 = df[df['Коды отказа'] == '01']
    # df_user_r091 = df[df['Коды отказа'].str.startswith('091', na=False)]
    df_vip = df[df['Статус заявки'] == 5]  # 'Заявка выпущена'

    title = 'КИ и Скоринговые балы'
    print(df.columns)
    return

    x_l = 'Сумма Скорингов'
    y_l = 'Сумма КИ'
    ss = ((df, plt.cm.binary, 'всего'),
          (df_appr, plt.cm.Blues, 'одобрены'),
          (df_rej, plt.cm.Reds, 'отклонены'),
          (df_user_r01091, plt.cm.Oranges, 'отказ клиента 091 01'),
          (df_vip, plt.cm.Greens, 'выпущены'))

    for s in ss:
        k, v, t = s
        x = k[x_l].fillna(0).to_numpy().flatten()
        y = k[y_l].fillna(0).astype(int).to_numpy().flatten()
        griddata_plot(x, y, 'griddata ' + title + ' ' + t, x_l, y_l, v)
        _plot_2dgist_and_two_gists_asone(x, y, title + ' 2дгист ' + t, x_l, y_l, v)


def plot_boxes(p):
    df: pd.DataFrame = pd.read_pickle(p)

    s_field = ['Запрошенная сумма кредита', 'Эквифакс 4Score']

    df = df[df['first_decision_state'] == 0]  # approved only
    df = df[s_field]

    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    for c in numerical_columns:
        df[c].plot.hist(bins=40)
        plt.title(c)
        # plt.savefig(c)
        plt.show()
        plt.close()


def plot_kde_plot_matrix(p):
    """ seaborn grid pairplot"""
    df: pd.DataFrame = pd.read_pickle(p)
    print(df.columns)
    print(df.select_dtypes(exclude=["object"]).columns)
    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    print(numerical_columns)
    # return
    sel_numeric = ['Запрошенная сумма кредита',
                   'Возраст клиента',
                   'Анкетный скоринг',
                   'Скоринговый балл НБКИ общ',
                   'Скоринговый балл ОКБ, основной скоринг бюро',
                   'Эквифакс 4Score',
                   'Сумма Скорингов']  # 'Месяц создания заявки', 'Час создания заявки',

    sel_cat = ['Тип машины', 'Оценка кредитной истории ОКБ',
               'Оценка кредитной истории НБКИ', 'Оценка кредитной истории Эквифакс', 'Сумма КИ', 'Час создания заявки',
               'Месяц создания заявки', 'Мегафон']

    s_field = ['Запрошенная сумма кредита', 'Анкетный скоринг',
        'Возраст клиента', 'Скоринговый балл НБКИ общ',
       'Скоринговый балл ОКБ, основной скоринг бюро', 'Эквифакс 4Score']
    df = df[df['first_decision_state'] == 0]
    df = df[sel_numeric]
    # print(df['Сумма Скорингов'].describe())
    # return
    for c in df.columns:
        print(c)
        df[c] = df[c].fillna(0) #.astype(int)

    # sns.pairplot(df, diag_kind='kde')

    # print(df.columns)
    g = sns.PairGrid(df)
    # g.map_diag(sns.kdeplot)
    g.map_upper(sns.histplot)
    g.map_lower(sns.histplot)
    # g.map_lower(sns.kdeplot)
    # plt.savefig('seaborn_diag_skd_scatter+kde')
    # g = sns.FacetGrid(df, palette="husl")  # hue="species" # col="variable" # col_wrap=2,
    # g.map(sns.kdeplot, "value", shade=True)
    # plt.legend(loc='upper right')
    plt.show()


def plot_one_box(p):
    """
    Etap 1
    :param p:
    :return:
    """

    df: pd.DataFrame = pd.read_pickle(p)
    from matplotlib.axes import Axes
    from scipy.interpolate import make_interp_spline

    # numerical_columns = df.select_dtypes(exclude=["object"]).columns
    # print(df.select_dtypes(include=["object"]).columns)
    # print(numerical_columns)
    # return
    # print(df[df['Статус заявки'] == 'Заявка выпущена']['Скоринговый балл ОКБ, основной скоринг бюро'].isin([0]).sum())
    #
    # print(df[df['Статус заявки'] == 'Заявка выпущена'].shape)
    # print(df[df['Статус заявки'] == 'Заявка выпущена']['Скоринговый балл ОКБ, основной скоринг бюро'].describe())
    # df[df['Статус заявки'] == 'Заявка выпущена'].hist()
    # plt.show()
    # return
    # print(df['Скоринговый балл ОКБ, основной скоринг бюро'].isna().sum())
    # print(df[df['Статус заявки'] == 'Заявка выпущена']['Скоринговый балл ОКБ, основной скоринг бюро'].isna().sum())
    # print(df[df['Статус заявки'] == 'Заявка выпущена']['Скоринговый балл ОКБ, основной скоринг бюро'].isin([0]).sum())
    #
    # print(df[df['Статус заявки'] == 'Заявка выпущена'].shape)
    # print(df['Статус заявки'].isna().sum())
    # print(df[df['Статус заявки'] == 'Заявка выпущена']['Скоринговый балл ОКБ, основной скоринг бюро'].describe())
    # exit()

    sel_numeric = ['Запрошенная сумма кредита',
                   'Возраст клиента',
                   'Анкетный скоринг',
                   'Скоринговый балл НБКИ общ',
                   'Скоринговый балл ОКБ, основной скоринг бюро',
                   'Эквифакс 4Score',
                   'Сумма Скорингов',
                   'B']  # 'Месяц создания заявки', 'Час создания заявки',

    sel_cat = ['Тип машины', 'Оценка кредитной истории ОКБ',
       'Оценка кредитной истории НБКИ', 'Оценка кредитной истории Эквифакс', 'Сумма КИ', 'Час создания заявки',
               'Месяц создания заявки', 'Мегафон']  # 'Коды отказа', 'МБКИ'

    df_n = df[sel_numeric + ['first_decision_state', 'Коды отказа', 'Статус заявки', 'ander']]

    # print(numerical_columns)
    # return
    # print(df['Эквифакс 4Score'].unique())
    df_n.fillna(-1, inplace=True)

    for c in sel_numeric:
        # continue
        # df_n[c] = df_n[df_n[c] != 0][c] #.astype(int)
        if df_n[c].hasnans:
            print(c, "WARNING")
            return
            continue
        # df_appr = df_n[(df_n['first_decision_state'] == 0) &
        #                (df_n['Коды отказа'] != '01') & ~(df_n['Коды отказа'].str.startswith('091', na=False))]  # 0 - 'approved'
        # df_rej = df_n[df_n['first_decision_state'] == 1]  # 'rejected'
        # df_user_r01091 = df_n[(df_n['Коды отказа'] == '01') | df_n['Коды отказа'].str.startswith('091', na=False)]
        df_appr = df_n[df_n['ander'] == 0]  # 0 - 'approved'
        df_rej = df_n[df_n['ander'] == 1]  # 'rejected'
        df_user_r01091 = df_n[df_n['ander'] == 2]
        # df_user_r01 = df_n[df_n['Коды отказа'] == '01']
        # df_user_r091 = df_n[df_n['Коды отказа'].str.startswith('091', na=False)]
        # TODO: выпущенные
        df_vip = df_n[df_n['Статус заявки'] == 5]  # 'Заявка выпущена'

        ax: Axe = None
        fig, ax = plt.subplots(2, 1, figsize=(19, 9))

        ax2 = ax[0].twinx()
        k = 2
        bins = 75
        if c == 'Час создания заявки':
            bins = 120
            k = 1
        elif c == 'Возраст клиента':
            bins = 40
            k = 1
        elif c == 'Анкетный скоринг':
            bins = 40
        elif c == 'Скоринговый балл НБКИ общ':
            bins = 30




        # всего серое
        y, binEdges = np.histogram(df_n[c].to_numpy(), bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        spline = make_interp_spline(bincenters, y, k=k)
        bincenters_sline = np.linspace(min(bincenters), max(bincenters), 200)
        y_new = spline(bincenters_sline)
        ax2.plot(bincenters_sline, y_new, '-k', label='все заявки')

        # отклонены красное
        y, binEdges = np.histogram(df_rej[c].to_numpy(), bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        spline = make_interp_spline(bincenters, y, k=k)
        bincenters_sline = np.linspace(min(bincenters), max(bincenters), 200)
        y_new = spline(bincenters_sline)
        ax2.plot(bincenters_sline, y_new, '-r', label='отклоненные (без 091, 01)')

        # y = y / df_appr.shape[0]
        # print(np.round(y * 10))
        # # return
        # new_appr = np.repeat(binEdges[:-1], y.astype(int))
        # print(df_appr.shape)
        # print(new_appr.shape)
        print(c)
        # d = df_rej[c].astype(int).replace(0, pd.NA)
        d = df_rej[c].to_numpy()
        d[d == 0] = np.nan
        d[d == -1] = np.nan
        if c == 'Скоринговый балл ОКБ, основной скоринг бюро':
            print(np.unique(d))
        # df_rej[c].replace(0.0, pd.NA)
        ax[1].hist(d, bins=bins, density=True, alpha=0.5, color='red', label='отклоненные (без 091, 01)')

        # синее акцептованные
        y, binEdges = np.histogram(df_appr[c].to_numpy(), bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        spline = make_interp_spline(bincenters, y, k=k)
        bincenters_sline = np.linspace(min(bincenters), max(bincenters), 200)
        y_new = spline(bincenters_sline)
        ax[0].plot(bincenters_sline, y_new, '-b', label='акцептованные')

        # d = df_appr[c].astype(int).replace(0.0, pd.NA)
        # df_appr[c].replace(0.0, pd.NA)
        d = df_appr[c].to_numpy()
        d[d == 0] = np.nan
        d[d == -1] = np.nan
        ax[1].hist(d, bins=bins, density=True, alpha=0.5, color='blue', label='акцептованные')

        #  зеленое выпущенные
        y, binEdges = np.histogram(df_vip[c].to_numpy(), bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        spline = make_interp_spline(bincenters, y, k=k)
        bincenters_sline = np.linspace(min(bincenters), max(bincenters), 200)
        y_new = spline(bincenters_sline)
        ax[0].plot(bincenters_sline, y_new, '-g', label='выпушенные')

        # df_vip[c].replace(0, pd.NA)
        # ax[1].hist(df_vip[c], bins=bins, density=True, alpha=0.5, color='green')

        y, binEdges = np.histogram(df_user_r01091[c].to_numpy(), bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        spline = make_interp_spline(bincenters, y, k=k)
        bincenters_sline = np.linspace(min(bincenters), max(bincenters), 200)
        y_new = spline(bincenters_sline)
        ax[0].plot(bincenters_sline, y_new, '-m', label='091 и 01 код отказа')

        ax[0].set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        if c == 'Скоринговый балл ОКБ, основной скоринг бюро':
            ax[0].set_xlim(left=500)
            ax[1].set_xlim(left=500)
            ax[0].set_ylim(top=600)
            ax2.set_ylim(top=1900)
        elif c == 'Эквифакс 4Score':
            ax[0].set_xlim(left=500)
            ax[1].set_xlim(left=500)
            ax2.set_ylim(top=2900)
        elif c == 'Скоринговый балл НБКИ общ':
            ax[0].set_xlim(left=400)
            ax[1].set_xlim(left=400)
            ax[0].set_ylim(top=1750)
            ax2.set_ylim(top=10000)
        elif c == 'Скоринговый балл ОКБ, основной скоринг бюро':
            ax[0].set_xlim(left=600)
            ax[1].set_xlim(left=600)
            ax[0].set_ylim(top=400)
            ax2.set_ylim(top=1750)
        elif c == 'Запрошенная сумма кредита':
            ax[0].set_xlim(left=500000)
            ax[1].set_xlim(left=500000)
            ax[0].set_xlim(right=2600000)
            ax[1].set_xlim(right=2600000)
            ax2.set_ylim(top=9000)
        elif c == 'Возраст клиента':
            ax[0].set_xlim(left=19)
            ax[1].set_xlim(left=19)
        elif c == 'Анкетный скоринг':
            ax[0].set_xlim(left=40)
            ax[1].set_xlim(left=40)
        elif c == 'Сумма Скорингов':
            ax[0].set_xlim(left=250)
            ax[1].set_xlim(left=250)
            ax2.set_ylim(top=3500)

        ax[0].legend(loc="upper left")
        ax[1].legend(loc="upper left")
        ax2.legend()

        plt.title(c)
        plt.savefig(c)
        # plt.show()
        # ax.remove)
        plt.close('all')

    # -- КАТЕГОРИАЛЬНЫЕ

    df2 = df
    for i, c_name in enumerate(sel_cat):
        # no nan
        df = df2.dropna(subset=[c_name])
        # remove Ошибка..
        c_with_b = ['Оценка кредитной истории ОКБ',
                    'Оценка кредитной истории НБКИ', 'Оценка кредитной истории Эквифакс']
        if c_name in c_with_b:
            df = df[(df[c_name] != 'Произошла ошибка') & (df[c_name] != 'Ошибка выполнения проверки')]

        if c_name == 'Мегафон':
            df = df[df[c_name] != -1]

        # sort
        x_l = list(df[c_name].unique())
        types = [type(v) for v in x_l]
        most_common = max(set(types), key=types.count)
        if most_common != str:
            x_l.sort()

        items = x_l
        # to strings
        x_l = [str(v) for v in x_l]

        print(c_name, x_l)
        print(c_name, items)

        df_appr = []
        df_rej = []
        df_user_r01091 = []
        df_vip = []
        df_ob = []
        for uv in items:
            df_appr.append(df[(df['first_decision_state'] == 0) & (df[c_name] == uv)].shape[0])  # 0 - 'approved'
            df_rej.append(df[(df['first_decision_state'] == 1) & (df[c_name] == uv)].shape[0])  # 'rejected'
            df_user_r01091.append(df[(df[c_name] == uv) & ((df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False))].shape[0])
            df_vip.append(df[(df['Статус заявки'] == 5) & (df[c_name] == uv)].shape[0])  # 'Заявка выпущена'
            df_ob.append(df[df[c_name] == uv].shape[0])

        fig, ax = plt.subplots(figsize=(19, 9))
        width = 0.13
        ax2 = ax.twinx()
        x = np.arange(0, len(x_l))

        ax.bar(x, df_appr, width=width, color='blue')
        ax.bar(x + width, df_user_r01091, width=width, color='orange')
        ax.bar(x + width * 2, df_vip, width=width, color='green')
        ax2.bar(x + width * 3, df_ob, width=width, color='black')
        ax2.bar(x + width * 4, df_rej, width=width, color='red')

        print(x_l)
        plt.xticks(x, x_l, rotation='vertical')

        title = "Количество заявок для " + c_name
        plt.title(title)
        ax.legend(['акцептованные', '091 и 01 код отказа', 'выпушенные'], loc="upper left")
        ax2.legend(['все заявки', 'отклоненные (без 091, 01)'], loc="upper right")
        # plt.show()
        plt.savefig(title)
        plt.close()


def pandas_prepare_unique_values_for_hist(df: pd.DataFrame, column: str) -> [list, list, list] or None:
    """

    :param df:
    :param column:
    :return: ( items, - unique values
    x_l, strings
    x, 0-n of uniq values
    ) or None
    """
    # sort
    x_l = list(df[column].unique())
    types = [type(v) for v in x_l]
    if len(types) != 0:
        most_common = max(set(types), key=types.count)
    else:
        print("Warning: no unique")
        return None
    if most_common != str:
        x_l.sort()

    items = x_l  # unique values of columns with NAN
    # to strings
    x_l = [str(v) for v in x_l]  # string of unique values

    x = np.arange(0, len(x_l))  # 0-10 of unique values
    return items, x_l, x


def plot_numerical_and_categorical(p, title:str = "hist and bars"):

    df: pd.DataFrame = pd.read_pickle(p)
    # df = p
    from matplotlib.axes import Axes
    from scipy.interpolate import make_interp_spline

    df_appr = df[df['ander'] == 0]  # 0 - 'approved'
    df_rej = df[df['ander'] == 1]  # 'rejected'
    df_user_r01091 = df[df['ander'] == 2]
    df_appr.drop('ander', axis=1, inplace=True)
    # return df_appr.shape[0]
    df_rej.drop('ander', axis=1, inplace=True)
    df_user_r01091.drop('ander', axis=1, inplace=True)
    # print(df_rej['Запрошенная сумма кредита'])
    # return
    print("groups", df_appr.shape[0], df_rej.shape[0], df_user_r01091.shape[0])
    # return

    numerical_columns = df_rej.select_dtypes(exclude=["object"]).columns.values
    categorical_columns = df_rej.select_dtypes(include=["object"]).columns.values.tolist()
    print(len(numerical_columns))
    print(len(categorical_columns))

    numerical_columns_1 = [c for c in numerical_columns if df[c].unique().shape[0] > 15]
    numerical_columns_2 = [c for c in numerical_columns if df[c].unique().shape[0] <= 15]
    print("numerical_1", len(numerical_columns_1))
    print("numerical_2", len(numerical_columns_2))
    other_columns = numerical_columns_2 + categorical_columns

    # df.fillna(-1, inplace=True)
    ax: Axes
    col_count = df.columns.values.shape[0] - 1
    print("total", col_count)

    cols = 6
    import math
    # print(col_count)
    # print(cols)
    rows = math.ceil(col_count / cols)
    # print(rows)
    # return
    fig, ax = plt.subplots(rows, cols, figsize=(19, 10))
    plt.subplots_adjust(left=0.076, right=0.96, bottom=0.04, top=0.96, wspace=0.30, hspace=0.85)  # hspace = wspace ||
    print(len(numerical_columns_1))
    print(len(other_columns))
    # return

    for i in range(rows):
        for j in range(cols):
            # if i > 1 or j >2:
            #     break
            n1 = i * cols + j
            n2 = n1 - len(numerical_columns_1)
            # print(i, j, n1)
            if n1 < len(numerical_columns_1):
                # -- NUMERICAL
                c = numerical_columns_1[n1]
                df_rej_2 = df_rej[df[c] != 0].copy()
                print(df_rej_2[c])
                df_user_r01091_2 = df_user_r01091[df[c] != 0].copy()
                df_appr_2 = df_appr[df[c] != 0].copy()

                print(i * rows + j, c)
                ax2 = ax[i][j].twinx()
                ax2.hist(df_rej_2[c], bins=15, color='red', alpha=0.5, label='отклоненные')
                ax[i][j].hist(df_user_r01091_2[c], bins=15, color='yellow', alpha=0.5, label='091 и 01')
                ax[i][j].hist(df_appr_2[c], bins=15, color='green', alpha=0.5, label='одобренные')
                ax[i][j].set_title(c, pad=15 * (j % 2))
            elif n2 < len(other_columns):
                # continue
                print("categorical", n2)
                c = other_columns[n2]

                # -- CATEGORICAL
                ret = pandas_prepare_unique_values_for_hist(df, c)
                # if c == 'Сумма Скорингов':
                #     print(df['Сумма Скорингов'].unique())
                #     print(ret)
                #     exit(0)

                if ret is None:
                    print("ERROR")
                    ax[i][j].set_title(c, pad=15 * (j % 2))
                    continue
                items, x_l, x = ret
                width = 0.13
                # print(x)
                # prepare data
                l_appr = []
                l_rej = []
                l_user_r01091 = []

                for uv in items:
                    l_appr.append(df_appr[df_appr[c] == uv].shape[0])  # 0 - 'approved'
                    l_rej.append(df_rej[df_rej[c] == uv].shape[0])  # 'rejected'
                    l_user_r01091.append(df_rej[df_rej[c] == uv].shape[0])
                print(len(l_rej), l_rej)

                ax[i][j].bar(x, l_appr, width=width, color='green')
                ax2 = ax[i][j].twinx()
                ax2.bar(x + width, l_rej, width=width, color='red')
                ax2.bar(x + width * 2, l_user_r01091, width=width, color='yellow')

                # plt.xticks(x, x_l, ax=ax[i][j], rotation='vertical')
                ax[i][j].set_xticks(x, minor=False)  # , x_l
                ax[i][j].set_xticklabels(x_l)  # , x_l
                ax[i][j].set_title(c, pad=15 * (j % 2))

    # show legend:
    ax[rows - 1][cols - 1].hist([], bins=15, color='yellow', alpha=0.6, label='091 и 01')
    ax[rows - 1][cols - 1].hist([], bins=15, color='green', alpha=0.6, label='одобренные')
    ax[rows - 1][cols - 1].hist([], bins=15, color='red', alpha=0.6, label='отклоненные (шкала справа)')
    ax[rows - 1][cols - 1].legend()
    # counts
    ax[rows - 1][cols - 2].hist([], bins=15, color='yellow', alpha=0.6, label=str(df_user_r01091.shape[0]))
    ax[rows - 1][cols - 2].hist([], bins=15, color='green', alpha=0.6, label=str(df_appr.shape[0]))
    ax[rows - 1][cols - 2].hist([], bins=15, color='red', alpha=0.6, label=str(df_rej.shape[0]))
    ax[rows - 1][cols - 2].legend()
    
    # plt.title('wf')
    plt.savefig(title + '.png')
    # plt.show()

