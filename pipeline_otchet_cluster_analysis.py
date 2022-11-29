import pandas as pd
import numpy as np
# own


def print_num_obj_cols(df: pd.DataFrame):
    categorial_columns = df.select_dtypes(include=["object"]).columns
    numerical_columns = df.select_dtypes(exclude=["object"]).columns
    print("numerical:")
    print(df[numerical_columns].describe().to_string())
    print("categorical:")
    print(df[categorial_columns].describe().to_string())
    # for c in categorial_columns:
    #     print(c, df[c].unique())


def csv_file_read(p):
    # -- get main table
    df = pd.read_csv(p, index_col=0, low_memory=False)
    df = df.sort_values(by=['Дата создания заявки'], ascending=False)

    # -- get status target
    df_and: pd.DataFrame = pd.read_pickle('deal_id_ander.pickle')
    df_and.set_index('id заявки', verify_integrity=True, inplace=True)
    df = df.join(df_and)
    df['ander'] = df['first_decision_state']

    df.loc[(df['ander'] == 'approved') &
           ((df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)), 'ander'] = 'user_rej_appr'
    df.loc[(df['ander'] == 'rejected') &
           ((df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)), 'ander'] = 'user_rej_rej'

    # print(df[df['ander'] == 'approved']['Коды отказа'].unique())


    # df = df[(df['Дата создания заявки'] > '2019-11-20') & (df['Дата создания заявки'] < '2021-04-28')]
    df = df[(df['Дата создания заявки'] > '2020-01-11') & (df['Дата создания заявки'] < '2021-01-01')]
    print('Одобрено', df[df['first_decision_state'] == 'approved'].shape)
    print('Отмена', df[df['first_decision_state'] == 'rejected'].shape)
    # df = df[df['Дата создания заявки'] >= '2020-01-03']
    # df = df[df['Дата создания заявки'] == '2020-01-14']
    # print(df)

    # print(df.columns)
    print('Одобрено', df[df['ander'] == 'approved'].shape)
    print('Отмена', df[df['ander'] == 'rejected'].shape)

    print('Откланено клиентом и андером', df[df['ander'] == 'user_rej_rej'].shape)
    print('Откланено клиентом, одобрено андером', df[df['ander'] == 'user_rej_appr'].shape)
    # df[df['ander'] == 'rejected'].to_csv('otchet.csv')
    exit(0)
    # print(df[df['ander'] == 'approved'])
    # print('код отмены 01 091',
    #       df[(df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)].shape)
    # print("всего", df.shape)
    # df = df[(df['ander'] == 'approved') | (df['ander'] == 'rejected')]
    # print('Отмена', df[df['ander'] == 'rejected'].shape)
    # print((df.isnull().sum() / df.shape[0] * 100).to_csv('a.csv'))
    # print(df[df['Дата создания заявки'] == '2020-01-03'])

    exit(0)
    # df = df[df['first_decision_state'] == 'approved']
    # print(df.shape)
    # return
    # df = df[df['first_decision_state'] == 'rejected']
    # print(df[df['ander'] == 'approved'].shape)
    # print(df[df['ander'] == 'approved'].tail(4000).to_csv("одобрено.csv"))
    # exit(0)

    # print(df[(df['Коды отказа'] == '01') | df['Коды отказа'].str.startswith('091', na=False)]['ander'])
    # return
    # --  выгрузка кодов отказа по популярности для rejected
    # a = df[df['first_decision_state'] == 'rejected'][['Коды отказа', 'Описание кодов отказа']]
    # a = pd.DataFrame(a.groupby(['Коды отказа', 'Описание кодов отказа']).size().reset_index(name="count"))
    # a = pd.DataFrame(a)
    # c_row = a.pop('count')
    # a.insert(0, 'count', c_row)
    # a.sort_values(by=['count'], ascending=False).to_csv('kod_otkaza.csv')
    # return
    # shift columnt to begining
    # target = df.pop('first_decision_state')
    # df.insert(1, 'first_decision_state', target)

    # -- observe
    # print(df.tail(200).head(100).to_string())
    # df.to_csv("/home/u2/evseeva/processed.csv")

    # -- удаляем лишние строки
    # df = df[df['first_decision_state'] == 'approved']
    # print(df.shape)
    # return
    # df = df[df['first_decision_state'] == 'rejected']
    # filter 091 and 01
    # df = df[df['Коды отказа'] != '01']
    # df091 = df[df['Коды отказа'].str.startswith('091', na=False)]
    # df = df.drop(index=df091.index)

    # -- Удаляем лишние столбцы
    l_col = [  # 'first_decision_state',  # одно значение
        'ФИО клиента',
        'id клиента',
        'ИНН работодателя',
        'Сделка дошла до Андерайтреа',  # одно значение
        'СФ андерайтера',  # после цели
        'Комментарии андерайтера',  # после цели
        # 'Коды отказа',  # после цели
        'Описание кодов отказа',  # дублирует код отказа
        'Коды системы',  # после цели -  заполняются Андерами.
        'Описание кодов',  # дублирует код системы
        'СФ системы',
        # Системная проверка - это проверки по которым сделка ушла в отказ. Они могут установится как до решения так и после решения Андерайтера
        'Системная проверка давшая СФ',
        # Системная проверка - это проверки по которым сделка ушла в отказ. Они могут установится как до решения так и после решения Андерайтера
        'Статус заявки',  # после цели
        'Решение по заявке',  # после цели
        'ФИО АНД принявшего последнее решение',
        'Дата создания заявки',  # дублирует Дата и Время создания заявки
        'Подтвержденная сумма кредита',  # после цели
        'Город Точки продажи',
        'Тип точки продаж',
        'Холдинг',
        'Наименование работодателя',
        'Доход клиента'
    ]

    # df.drop(l_col, axis=1, inplace=True)

    c_select = ['Дата и Время создания заявки',
                'Дата рождения клиента', 'Запрошенная сумма кредита', 'Тип машины',
                'Коды отказа', 'Мегафон', 'МБКИ',
                'Скоринговый балл НБКИ Digital Score',
                'Скоринговый балл ОКБ, основной скоринг бюро',
                'Оценка кредитной истории ОКБ', 'Скоринговый балл НБКИ FiCO',
                'Оценка кредитной истории НБКИ', 'Эквифакс 4Score',
                'Оценка кредитной истории Эквифакс', 'Анкетный скоринг',

                'Статус заявки',
                'first_decision_state',
                'ander']
    df = df[c_select]
    # print(df.tail(200).head(100).to_string())
    # print(df['Коды системы'].unique())

    # -- форматирование столбцов
    # print(df['Оценка кредитной истории ОКБ'].unique())
    # print(df['Оценка кредитной истории Эквифакс'].unique().tolist())
    # print(df['Оценка кредитной истории НБКИ'].unique())

    # before: 'Хорошая КИ (отчет: https://api-v2.prod.norma.rnb.com/bki-reports/nbki/206a6074930314431a832a0aca2df20d.pdf)'
    # after 'Хорошая КИ '
    http_dirt_cols = [
        'Оценка кредитной истории ОКБ',
        'Оценка кредитной истории Эквифакс',
        'Оценка кредитной истории НБКИ'
    ]
    for c in http_dirt_cols:
        df[c] = df[c].str.extract(r"^([\w ]*)[^\(]?")
        df[c] = df[c].apply(lambda x: str.strip(x) if pd.notna(x) else x)

    # print(df[df['first_decision_state'] == 'approved'].groupby('Оценка кредитной истории Эквифакс').size().reset_index(
    #     name="count"))  # appr
    # # print(df[df['first_decision_state'] == 'approved'][df['Оценка кредитной истории Эквифакс'] == 'Хорошая КИ']['Оценка кредитной истории Эквифакс'].notna().sum())  # appr
    # print(df[df['first_decision_state'] == 'approved'][df['Оценка кредитной истории Эквифакс'] == 'Хорошая КИ'].tail(10))
    # print(df[df['first_decision_state'] == 'rejected'].groupby('Оценка кредитной истории Эквифакс').size().reset_index(
    #     name="count"))
    # print(df[df['first_decision_state'] == 'rejected'][
    #           df['Оценка кредитной истории Эквифакс'] == 'Хорошая КИ'].tail(10))  # appr
    # return
    # print(df.isna().sum())

    # нецифровых значения становятся ''
    df['Скоринговый балл НБКИ Digital Score'] = df['Скоринговый балл НБКИ Digital Score'].str.extract(r"^(\d*)")
    df['Скоринговый балл НБКИ FiCO'] = df['Скоринговый балл НБКИ FiCO'].str.extract(r"^(\d*)")
    # df['Коды отказа'] = df['Коды отказа'].str.extract(r"^(\d*)")
    df.dropna(0, subset=['ander'], inplace=True)
    print((df.isna().sum() / df.shape[0] * 100).to_string())
    return

    # -- корректируем типы
    # int
    int_col = ['Скоринговый балл НБКИ Digital Score',
               'Скоринговый балл ОКБ, основной скоринг бюро',
               'Скоринговый балл НБКИ FiCO',
               'Эквифакс 4Score',
               'Анкетный скоринг',
               # 'Коды отказа'
               ]

    for col in int_col:
        # 'coerce' - if error we set NaN
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # -- кодируем строки не требующие восполнения пропущенных
    # for c in df.columns:
    #     if df[c].dtype != 'object':
    #         print(c, df[c].isna().sum())
    # return
    # print(df.shape)
    # df_orig = df.copy()

    # print(dict(enumerate(df['first_decision_state'].astype('category').cat.categories)))
    print(dict(enumerate(df['Статус заявки'].astype('category').cat.categories)))
    print(dict(enumerate(df['first_decision_state'].astype('category').cat.categories)))
    print(dict(enumerate(df['ander'].astype('category').cat.categories)))
    print(dict(enumerate(df['Мегафон'].astype('category').cat.categories)))

    # {0: 'gos_fill_info', 1: 'recalc_payment_graphic', 2: 'rescoring_success', 3: 'Доработка по КИ',
    #  4: 'Доработко по окончателньому решению', 5: 'Заявка выпущена', 6: 'Заявка отменена', 7: 'Заявка пророчена',
    #  8: 'Клиент подписал', 9: 'На Андерайтере', 10: 'На доработке ГОЗ', 11: 'Новая заявка', 12: 'Окончателньое решение',
    #  13: 'Окончательное решение после предварительного', 14: 'Отправлена на выдачу', 15: 'Отправлена на выпуске',
    #  16: 'Ошибка при резерве счета', 17: 'Предварительное решение'}
    # {0: 'approved', 1: 'rejected'}
    # ander: {0: 'approved', 1: 'rejected', 2: 'user_rej'}
    # {
    #     0: 'cURL error 35: error:14094415:SSL routines:ssl3_read_bytes:sslv3 alert certificate expired (see https://curl.haxx.se/libcurl/c/libcurl-errors.html)',
    #     1: 'cURL error 7: Failed to connect to ml.datalab.megafon.ru port 10443: Connection refused (see https://curl.haxx.se/libcurl/c/libcurl-errors.html)',
    #     2: 'Данные по указанному номеру телефона не найдены',
    #     3: 'Клиенту отказано. С даты активации абонентского номера прошло - менее 30 дней',
    #     4: 'Клиенту отказано. С даты активации абонентского номера прошло - от 1 до 3 месяцев',
    #     5: 'Клиенту отказано. С даты активации абонентского номера прошло - от 3 до 6 месяцев',
    #     6: 'Клиенту отказано. С даты активации абонентского номера прошло - от 6 до 12 месяцев',
    #     7: 'Необходим легендированный звонок на работу или звонок по альтернативному рабочему телефону. С даты активации абонентского номера прошло - от 6 до 12 месяцев',
    #     8: 'Необходим легендированный звонок на работу. С даты активации абонентского номера прошло - более 3 лет',
    #     9: 'Необходим легендированный звонок на работу. С даты активации абонентского номера прошло - от 1 года до 3 лет',
    #     10: 'Необходим легендированный звонок на работу. С даты активации абонентского номера прошло - от 3 до 6 месяцев',
    #     11: 'Необходим легендированный звонок на работу. С даты активации абонентского номера прошло - от 6 до 12 месяцев',
    #     12: 'Нет ограничений', 13: 'Ошибка выполнения проверки',
    #     14: 'Проверка не выполнена. Первоначальный взнос больше 40% и/или класс заемщика не в допустимом диапазоне.',
    #     15: 'Точка продаж не соответсвует разрешенному списку.'}

    df['Мегафон'] = df['Мегафон'].astype('category').cat.codes
    df['first_decision_state'] = df['first_decision_state'].astype('category').cat.codes
    df['ander'] = df['ander'].astype('category').cat.codes
    df['Статус заявки'] = df['Статус заявки'].astype('category').cat.codes
    # print(df[df['Статус заявки'] == 'Заявка выпущена'].shape)

    # print(df['Статус заявки'].unique())
    # return
    # df['Коды отказа'] = df['Коды отказа'].map({''})
    # print(df['Коды отказа'].isna().sum())
    # print(df.shape)
    # return

    # print(df['Мегафон'].isna().sum())

    # -- создаем столбцы выделяя значимые данные
    # дата рождения
    df['Возраст клиента'] = pd.to_numeric(2021 - pd.to_datetime(df['Дата рождения клиента']).dt.year).astype(
        'Int32')
    # df['Месяц рождения клиента'] = pd.to_datetime(df['Дата рождения клиента']).dt.month
    df.drop('Дата рождения клиента', axis=1, inplace=True)
    df['Час создания заявки'] = pd.to_numeric(pd.to_datetime(df['Дата и Время создания заявки']).dt.hour).astype(
        'Int32')
    df['Месяц создания заявки'] = pd.to_numeric(pd.to_datetime(df['Дата и Время создания заявки']).dt.month).astype(
        'Int32')
    df['День недели'] = pd.to_numeric(pd.to_datetime(df['Дата и Время создания заявки']).dt.dayofweek).astype(
        'Int32')
    df.drop('Дата и Время создания заявки', axis=1, inplace=True)
    # df['подтв_минус_доход'] = df['Доход клиента'] - df['Подтвержденный доход клиента']

    # print(df.tail(200).head(10).to_string())
    # объединяем скоринговый бал
    # пороверяем средние

    # import matplotlib.pyplot as plt
    # df.boxplot(column=['Скоринговый балл НБКИ Digital Score', 'Скоринговый балл НБКИ FiCO'])
    # plt.show()
    # return
    # df['Скоринговый балл НБКИ Digital Score'].fillna(0, inplace=True)
    # df['Скоринговый балл НБКИ FiCO'].fillna(0,inplace=True)
    # a = df[(df['Скоринговый балл НБКИ Digital Score'] != 0) & (df['Скоринговый балл НБКИ FiCO'] != 0)]
    # print(a.describe().to_string())
    # return

    df['Скоринговый балл НБКИ общ'] = df['Скоринговый балл НБКИ Digital Score'].fillna(0) + df[
        'Скоринговый балл НБКИ FiCO'].fillna(0)
    df.drop('Скоринговый балл НБКИ Digital Score', 1, inplace=True)
    df.drop('Скоринговый балл НБКИ FiCO', 1, inplace=True)

    # print(df.tail(200).head(10).to_string())
    # print(df.columns)

    cred_ist1 = ['Оценка кредитной истории ОКБ',
                 'Оценка кредитной истории НБКИ',
                 'Оценка кредитной истории Эквифакс']
    scorings2 = [
        'Скоринговый балл ОКБ, основной скоринг бюро',
        'Эквифакс 4Score',
        'Скоринговый балл НБКИ общ',
        'Анкетный скоринг'
    ]
    # for c in scorings2:
    #     print(c, df[c].isna().sum())
    # return
    # df['Оценка кредитной истории'] = df[cred_ist1].sum()
    # df['Скоринги'] = df[scorings2].sum()
    # print(df['Анкетный скоринг'].describe())
    df_sc = df[scorings2].copy()
    for c in df_sc:
        df_sc[c].fillna(0, inplace=True)
        df_sc[c] = (df_sc[c] - df_sc[c].min()) / (df_sc[c].max() - df_sc[c].min())
    # df_sc = (df_sc - df_sc.mean(axis=0)) / df_sc.std(axis=0)

    #
    df['Сумма Скорингов'] = (df_sc['Скоринговый балл ОКБ, основной скоринг бюро'] + df_sc['Эквифакс 4Score']
                             + df_sc['Скоринговый балл НБКИ общ'] + df_sc['Анкетный скоринг']) * 300  # 1200
    # print(df['Сумма Скорингов'].describe())
    # df[df['first_decision_state'] == 0]['Скоринги'].hist(bins=40)
    # from matplotlib import pyplot as plt
    # plt.show()
    # exit()

    # print(df['Оценка кредитной истории ОКБ'].unique())
    # print(df['Скоринговый балл ОКБ, основной скоринг бюро'])
    # print(df['Оценка кредитной истории НБКИ'].unique())
    # print(df['Оценка кредитной истории Эквифакс'].unique())
    # print(dict(enumerate(df['Оценка кредитной истории ОКБ'].astype('category').cat.categories)))
    # print(dict(enumerate(df['Оценка кредитной истории НБКИ'].astype('category').cat.categories)))
    # print(dict(enumerate(df['Оценка кредитной истории Эквифакс'].astype('category').cat.categories)))
    # df['Оценка кредитной истории ОКБ'] = df['Оценка кредитной истории ОКБ'].astype('category').cat.codes
    # df['Оценка кредитной истории НБКИ'] = df['Оценка кредитной истории НБКИ'].astype('category').cat.codes
    # df['Оценка кредитной истории Эквифакс'] = df['Оценка кредитной истории Эквифакс'].astype('category').cat.codes
    repl = ((0, 'Плохая КИ'),
            (1, 'Произошла ошибка'),
            (1, 'Ошибка выполнения проверки'),
            (2, 'КИ отсутствует'),
            (3, 'Нейтральная КИ'),
            (4, 'Хорошая КИ'))

    df_c = df[cred_ist1].copy()
    for c in cred_ist1:
        for p in repl:
            k, v = p
            df_c[c] = df_c[c].replace(v, k)
        df_c[c].fillna(1, inplace=True)
    df_c = df_c.rename(columns={"Оценка кредитной истории ОКБ": "Оценка кредитной истории ОКБ^",
                         "Оценка кредитной истории НБКИ": 'Оценка кредитной истории НБКИ^',
                         'Оценка кредитной истории Эквифакс': 'Оценка кредитной истории Эквифакс^'})
    # NA + 3 = NA
    # print(df_c['Оценка кредитной истории ОКБ^'].unique())
    # print(df_c['Оценка кредитной истории НБКИ^'].unique())
    # print(df_c['Оценка кредитной истории Эквифакс^'].unique())
    df['Сумма КИ'] = (df_c['Оценка кредитной истории ОКБ^'] + df_c['Оценка кредитной истории НБКИ^'] + df_c[
        'Оценка кредитной истории Эквифакс^']) * 100
    df = pd.concat([df, df_c], axis=1, sort=False, verify_integrity=True)
    # print(df)
    # return

    # print(df.dtypes)
    # return

    # [print(repr(x)) for x in df['МБКИ'].unique().tolist()]
    # exit(0)

    # МБКИ
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

    df['МБКИ'] = df['МБКИ'].apply(a)  # clear errors
    import re

    df['МБКИ_адрес'] = df['МБКИ'].apply(lambda x:
                                        re.search('Требуется проверка адреса регистрации/фактического',
                                                  x) is not None
                                        if pd.notna(x) else False).astype(int)
    df['МБКИ_исп_пр'] = df['МБКИ'].apply(lambda x:
                                         re.search('Требуется проверка данных о наличии исполнительных про',
                                                   x) is not None
                                         if pd.notna(x) else False).astype(int)
    df['МБКИ_неогр'] = df['МБКИ'].apply(lambda x:
                                        re.search('Нет ограни',
                                                  x) is not None
                                        if pd.notna(x) else False).astype(int)
    df['МБКИ_недост'] = df['МБКИ'].apply(lambda x:
                                         re.search('Недостаточно данных для выполнения пр', x) is not None or
                                         re.search('Данные по клиенту не найд', x) is not None
                                         if pd.notna(x) else False).astype(int)
    df['МБКИ_розыск'] = df['МБКИ'].apply(lambda x:
                                         re.search('Лица, находящиеся в р',
                                                   x) is not None
                                         if pd.notna(x) else False).astype(int)
    df['МБКИ_невыполнена'] = df['МБКИ'].apply(lambda x:
                                              re.search('Проверка не выполнена',
                                                        x) is not None
                                              if pd.notna(x) else False).astype(int)
    df['МБКИ_спецуч'] = df['МБКИ'].apply(lambda x:
                                         re.search('Наличие информации о постановке клиента на спец',
                                                   x) is not None
                                         if pd.notna(x) else False).astype(int)
    df['МБКИ_паспорт'] = df['МБКИ'].apply(lambda x:
                                          re.search('Требуется проверка информации о действительности па',
                                                    x) is not None
                                          if pd.notna(x) else False).astype(int)
    # df['МБКИ'] = df['МБКИ_адрес'].astype(int)*0.317792 - df['МБКИ_исп_пр'].astype(int)*0.429246 + df['МБКИ_неогр'].astype(int)*0.180987
    # df['МБКИ'] = df['МБКИ_исп_пр'].map({True:-1, False:1}) + df[
    #     'МБКИ_неогр'].astype(int) + df['МБКИ_невыполнена'].astype(int) + df['МБКИ_недост'].map({True:0, False:1}) \
    #              + df['МБКИ_розыск'].map({True:0, False:1}) + df['МБКИ_адрес'].map({True:1, False:-1})
    # df['A'] = df['Сумма КИ'] + df['Эквифакс 4Score'] + df['Скоринговый балл ОКБ, основной скоринг бюро'] * 0.89
    df['B'] = df['Сумма КИ'] + df['Эквифакс 4Score'] + df['Сумма Скорингов'] * 0.5
    # print(df['МБКИ'].unique())
    # print(df['МБКИ_адрес'].unique())
    # drop source columns:
    # df.drop(['МБКИ'] + cred_ist1 + scorings2, axis=1, inplace=True)

    df = df.reset_index(drop=True)

    return save('after_read.pickle', df)


def select_columns(p):
    df: pd.DataFrame = pd.read_pickle(p)

    selected_columns = [
        'Скоринговый балл ОКБ, основной скоринг бюро',
        'Эквифакс 4Score',
        'Скоринговый балл НБКИ общ',
        'Анкетный скоринг',
        'Сумма КИ',
        'Сумма Скорингов',
        'Запрошенная сумма кредита',
        # 'A',
        # 'B',
        'Оценка кредитной истории ОКБ^',
        'Оценка кредитной истории НБКИ^',
        'Оценка кредитной истории Эквифакс^',
        'Оценка кредитной истории ОКБ',
        'Оценка кредитной истории НБКИ',
        'Оценка кредитной истории Эквифакс',
        # 'МБКИ',
        'Возраст клиента',
        'ander',
        'МБКИ_адрес',
        'МБКИ_исп_пр',
        'МБКИ_неогр',
        'МБКИ_недост',
        'МБКИ_розыск',
        'МБКИ_невыполнена'
    ]
    selected_columns_freq = [
        'Скоринговый балл ОКБ, основной скоринг бюро',
        'Эквифакс 4Score',
        'Скоринговый балл НБКИ общ',
        'Анкетный скоринг',
        # 'Сумма КИ',
        # 'Сумма Скорингов',
        'Запрошенная сумма кредита',
        # 'A',
        # 'B',
        'Оценка кредитной истории ОКБ^',
        'Оценка кредитной истории НБКИ^',
        'Оценка кредитной истории Эквифакс^',
        # 'Оценка кредитной истории ОКБ',
        # 'Оценка кредитной истории НБКИ',
        # 'Оценка кредитной истории Эквифакс',
        'ander']
    # df = df[selected_columns]
    # df = df[selected_columns_freq]

    sc = [
        # 'Сумма КИ',
        'Оценка кредитной истории ОКБ',
          # 'Оценка кредитной истории ОКБ^',
        #   'Эквифакс 4Score',
        # 'Оценка кредитной истории НБКИ',
        # 'Запрошенная сумма кредита',
        # 'B',
        # 'Оценка кредитной истории Эквифакс^',
        # 'Оценка кредитной истории Эквифакс',
          'ander']

    # print(df['ander'].unique())
    # print(df[df['ander'] == -1].shape[0])
    # exit(0)
    # df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    # print(df['ander'].unique())
    # print(df[df['ander'] == 0].shape[0])
    # print(df[df['ander'] == 1].shape[0])
    # print(df[df['ander'] == 2].shape[0])
    # df[df['ander'] == 2] = 1
    # print(df.isna().sum())

    df = df[sc]


    return save('columns_selected.pickle', df)


def feature_engeering(p, exclude: list = None, remove: list = None, selected_columns: list = None):
    import featuretools as ft
    df: pd.DataFrame = pd.read_pickle(p)
    # -- drop columns
    if remove is not None:
        df = df.drop(remove, 1)
    # -- reset index
    df = df.reset_index(drop=True)

    # remove save excluded columns
    if exclude is not None:
        df_exception = df[exclude].copy()
        df = df.drop(exclude, 1)

    # sel_columns
    if selected_columns is not None:
        df = df[selected_columns]

    es = ft.EntitySet()
    es.entity_from_dataframe(entity_id="df",
                             dataframe=df,
                             index="id")
    # print(es)
    # trans_primitives_default = ["year", "month", "weekday", "haversine"]
    # print(ft.primitives.list_primitives().to_string())

    trans_primitives = ['divide_numeric', 'add_numeric']  # ['add_numeric']
    # agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
    # primitive_options = {tuple(trans_primitives_default + agg_primitives): {'ignore_variables': ignore_variables}}

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="df",
                                          max_depth=3, verbose=True, n_jobs=-1,
                                          trans_primitives=trans_primitives,
                                          # primitive_options=primitive_options
                                          )  # MAIN

    feature_matrix: pd.DataFrame = feature_matrix
    # print(feature_matrix.head().to_string())
    df = feature_matrix
    df.replace(np.inf, 999999999, inplace=True)
    df.replace(-np.inf, -999999999, inplace=True)
    df.fillna(0, inplace=True)

    # return
    if exclude is not None:
        df = pd.concat([df, df_exception], axis=1, sort=False, verify_integrity=True)

    print(df.head().to_string())

    p = 'feature_eng.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def forest_search_parameters(p, target: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import cross_val_score
    kfold = StratifiedKFold(n_splits=3)

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop([target], 1)
    Y = df[target]
    params = {'n_estimators': [2, 6], 'min_samples_split': [2, 3],
              'max_leaf_nodes': list(range(2, 15)), 'max_depth': list(range(2, 10))}
    clf = GridSearchCV(RandomForestClassifier(), params, cv=kfold)
    results = clf.fit(X, Y)
    # min_samples_split - will not be displayed
    print(results.best_estimator_)
    results = cross_val_score(results.best_estimator_, X, Y, cv=2)
    print("Accuracy: %f" % results.mean())


def feature_importance_forest2(p: str, target: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import cross_val_score

    params = {'n_estimators': [2, 4], 'min_samples_split': [2, 3],
              'max_leaf_nodes': list(range(2, 10)), 'max_depth': list(range(2, 5))}

    df: pd.DataFrame = pd.read_pickle(p)
    X = df.drop([target], 1)
    Y = df[target]

    importance_sum = np.zeros(X.shape[1], dtype=float)

    for i in range(2, 5):
        for j in range(2, 5):
            kfold = StratifiedKFold(n_splits=i)

            clf = GridSearchCV(RandomForestClassifier(), params, cv=kfold)
            results = clf.fit(X, Y)
            # min_samples_split - will not be displayed
            print(results.best_estimator_)
            results2 = cross_val_score(results.best_estimator_, X, Y, cv=j)
            print("Accuracy: %f" % results2.mean())

            model = results.best_estimator_
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


def feature_importance_forest(p: str, target: str, max_depth=12, n_estimators=25, max_leaf_nodes=14,
                              min_samples_split=2):
    """
    :param df:
    :param max_depth:
    :param n_estimators:
    :param max_leaf_nodes:+5
    """
    df: pd.DataFrame = pd.read_pickle(p)
    # df.drop(['first_decision_state', 'Коды отказа', 'Статус заявки'], axis=1, inplace=True)

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

    X = df.drop([target], 1)
    Y = df[target]

    from sklearn.ensemble import RandomForestClassifier

    importance_sum = np.zeros(X.shape[1], dtype=float)
    n = 100
    max_depth = np.linspace(2, max_depth + 8, 100)  # 12
    n_estimators = np.linspace(2, n_estimators + 15, 100)  # 25
    max_leaf_nodes = np.linspace(max_leaf_nodes - 4, max_leaf_nodes + 8, 100)  # 14

    for i in range(n):
        depth = int(round(max_depth[i]))
        n_est = int(round(n_estimators[i]))
        max_l = int(round(max_leaf_nodes[i]))

        model = RandomForestClassifier(random_state=i, max_depth=depth,
                                       n_estimators=n_est, max_leaf_nodes=max_l,
                                       min_samples_split=min_samples_split)
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

    # def corr_matrix(p: str):
    import seaborn
    import matplotlib.pyplot as plt

    # # print(df.columns.values)
    # df.corr(method="spearman").to_csv('corr_matrix.csv')
    # return
    # seaborn.heatmap(df.corr(), annot=True)
    # plt.show()


def hierarchical_clustering_filter(labels: list, p):
    df: pd.DataFrame = pd.read_pickle(p)
    print(df.shape)
    print("clasters count " + str(len(set(labels))))
    assert df.shape[0] == len(labels)
    # add labels field
    df['labels'] = labels
    # select clasters which has > 1 elements
    selected_labels = []
    for i in set(labels):
        if list(labels).count(i) > 2:
            selected_labels.append(i)
    df = df[df['labels'].isin(selected_labels)]
    print(df.shape)

    p = 'cluster_filter.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def k_mean(p):
    df: pd.DataFrame = pd.read_pickle(p)
    from sklearn.cluster import KMeans


def frequency_analysis(p):
    df: pd.DataFrame = pd.read_pickle(p)
    from matplotlib import pyplot as plt
    # print(df.tail(100).to_string())
    # return
    # [print(c) for c in df.columns.to_list()]
    # return
    # print(df['first_decision_state'].describe())
    # return
    # print(df['Скоринговый балл ОКБ, основной скоринг бюро'].isna().sum())
    # print(df.shape)
    # print(df[df['Статус заявки'] == 5]['Скоринговый балл ОКБ, основной скоринг бюро'].isna().sum())  # 'Заявка выпущена'
    # print(df[df['Статус заявки'] == 5]['Скоринговый балл ОКБ, основной скоринг бюро'].isin([0]).sum())
    # print(df[df['Статус заявки'] == 5].shape)
    # print(df['Статус заявки'].isna().sum())
    # print(df[df['Статус заявки'] == 5]['Скоринговый балл ОКБ, основной скоринг бюро'].describe())

    # categorial_columns = df.select_dtypes(include=["object"]).columns
    # numerical_columns = df.select_dtypes(exclude=["object"]).columns
    # print(categorial_columns)
    # print(numerical_columns)
    # return
    # print((df[df['first_decision_state'] == 0].isna().sum()/df[df['first_decision_state'] == 0].shape[0]) /
    #       df[df['first_decision_state'] == 1].isna().sum()/df[df['first_decision_state'] == 1].shape[0])
    # print(df[df['first_decision_state'] == 1].isna().sum()/df[df['first_decision_state'] == 1].shape[0])
    print(df.shape[0] - 2000)
    df_0 = df[df['ander'] == 0]  # appr
    df_1 = df[df['ander'] == 1]  # rej without 091
    print("appr, rej", df_0.shape[0], df_1.shape[0])
    print("NA", df_1.isna().sum())
    # print(df_0.shape[0], df_1.shape[0])
    # return
    # df = df.tail(df.shape[0] - 3000)
    other_c = []
    for i, c in enumerate(df.columns):
        # if i> 10:
        #     break
        if df[c].unique().shape[0] > 40:
            other_c.append(c)
            continue
        df_c_0 = df_0.groupby(c).size().reset_index(
            name="Одобренных")
        df_c_1 = df_1.groupby(c).size().reset_index(
            name="Отклоненных")
        # print("1", df_c_0['count__0'])
        # print(df_c_1['count'])
        df_c_0['count'] = df_c_0["Одобренных"] / df_0.shape[0]
        df_c_1['count'] = df_c_1["Отклоненных"] / df_1.shape[0]
        # print(df_c_0)  # значения и частота принятые # approved
        # print(df_c_1)  # значения и частота отвергнутые
        df_c_0.set_index(c, verify_integrity=True, inplace=True)
        df_c_1.set_index(c, verify_integrity=True, inplace=True)
        df_c = df_c_0.join(df_c_1, lsuffix='_0', rsuffix='_1')
        df_c['Отношение одоб/откл'] = round(np.log(df_c['count_0'] / df_c['count_1']), 5)
        df_c.drop(['count_0', 'count_1'], 1, inplace=True)  # del used columns
        print(df_c.to_markdown())
        with open('a.csv', 'a') as f:
            df_c.to_csv(f)
            f.write('\n')
        # print(df_c.to_string())
        # print(df[df['first_decision_state'] == 1].isna().sum()[c])
        # print(df[df['first_decision_state'] == 0].isna().sum()[c])
        # print(df[df['first_decision_state'] == 0][c].shape[0], df[df['first_decision_state'] == 0][df[c] != pd.NA].shape[0])
        # print(df[df['first_decision_state'] == 1][c].shape[0], df[df['first_decision_state'] == 1][df[c] != pd.NA].shape[0])
    print("Other columns:", other_c)
    # return
    #
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
    return


def add_sum_forest(p, last_n: int, a, bb, c, d, e, f, g):
    """ to rescale to 1 0 we requre encude all categprical columns to numbers"""
    df: pd.DataFrame = pd.read_pickle(p)

    # intro
    df = df.tail(df.shape[0] - 1000)
    df = df.reset_index(drop=True)
    # df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    df = df[(df['ander'] == 0) | (df['ander'] == 1)]
    # df[df['ander'] == 2] = 1

    # sum
    df2 = df.copy()
    n_min, n_max = 0, 1
    minimum, maximum = np.min(df, axis=0), np.max(df, axis=0)
    m = (n_max - n_min) / (maximum - minimum)
    b = n_min - m * minimum
    df2 = m * df2 + b
    df['sum'] = \
        (df2['Оценка кредитной истории ОКБ^'] * a +
         df2['Эквифакс 4Score'] * bb +
         df2['Оценка кредитной истории Эквифакс^'] * c +
         df2['Оценка кредитной истории НБКИ^'] * d +
         df2['Скоринговый балл ОКБ, основной скоринг бюро'] * e +
         df2['Анкетный скоринг'] * f +
         df2['Скоринговый балл НБКИ общ'] * g)

    # filter
    df = df.sort_values(by='sum')
    c = df[df['ander'] == 0].shape[0]
    df1 = df[df['ander'] == 0].tail(c - round(c/4))  # appr
    c = df[df['ander'] == 1].shape[0]
    df2 = df[df['ander'] == 1].head(c - round(c/4))  # rej
    df = pd.concat([df1, df2])


    # print(df[df['ander'] == 2].shape[0])
    # exit(0)

    # outro
    sc = [
        'Сумма КИ',
        'Оценка кредитной истории ОКБ^',
        'Эквифакс 4Score',
        'Оценка кредитной истории НБКИ^',
        'Запрошенная сумма кредита',
        'МБКИ',
        'B',
        'sum',
        'Оценка кредитной истории Эквифакс^',

        'ander'
        ]
    df = df[sc]

    print(df[df['ander'] == 0].shape[0])
    print(df[df['ander'] == 1].shape[0])
    print(df[df['ander'] == 0].shape[0] / df[df['ander'] == 1].shape[0])
    # exit(0)

    p = 'with_sum.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def add_sum(p, last_n: int, a, bb ,c ,d ,e, f, g):
# def add_sum(p, last_n: int):
    """ to rescale to 1 0 we requre encude all categprical columns to numbers"""
    df: pd.DataFrame = pd.read_pickle(p)
    # df = p
    # df['Скоринговый балл ОКБ, основной скоринг бюро'] + \
    # df['Эквифакс 4Score'] + \
    # df['Скоринговый балл НБКИ общ'] + \
    df = df.tail(df.shape[0] - 2000)
    df = df.reset_index(drop=True)
    df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]

    df2 = df.copy()
    n_min, n_max = 0, 1
    minimum, maximum = np.min(df, axis=0), np.max(df, axis=0)
    m = (n_max - n_min) / (maximum - minimum)
    b = n_min - m * minimum
    df2 = m * df2 + b

    # df['sum'] = \
    # df2['Сумма КИ'] + \
    # df2['Сумма Скорингов']
    # df['sum'] = \
    #     (df2['Оценка кредитной истории ОКБ^']* a +
    #      df2['Эквифакс 4Score'] * bb +
    #      df2['Оценка кредитной истории Эквифакс^'] * c +
    #      df2['Оценка кредитной истории НБКИ^'] * d +
    #      df2['Скоринговый балл ОКБ, основной скоринг бюро'] * e +
    #      df2['Анкетный скоринг'] * f +
    #      df2['Скоринговый балл НБКИ общ'] * g)
    df['sum'] = \
        (df2['Оценка кредитной истории ОКБ^'] * a +
         df2['Эквифакс 4Score'] * bb +
         df2['Оценка кредитной истории Эквифакс^'] * c +
         df2['Оценка кредитной истории НБКИ^'] * d +
         df2['Скоринговый балл ОКБ, основной скоринг бюро'] * e +
         df2['Анкетный скоринг'] * f +
         df2['Скоринговый балл НБКИ общ'] * g)

    # 'A',
    # 'B',
    # df['МБКИ']
    # 'Возраст клиента',
    # 'ander',
    # print(df['sum'].describe())
    df = df.sort_values(by='sum')
    # print(df.head(5).to_string())
    # df = df.tail(50)
    # print("shape", df.shape)
    df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    # print("shape", df.shape)
    df = df.tail(last_n)
    # print("shape", df.shape)


    p = 'with_sum.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p
    # return df


def add_sum_odob_rej(p, last_n: int = 2, last_n_slide=-1):
    """ to rescale to 1 0 we requre encude all categprical columns to numbers"""
    df: pd.DataFrame = pd.read_pickle(p)
    # df['Скоринговый балл ОКБ, основной скоринг бюро'] + \
    # df['Эквифакс 4Score'] + \
    # df['Скоринговый балл НБКИ общ'] + \
    # df2['Запрошенная сумма кредита'] + \
    # df2['Сумма КИ'] + \
    df = df.tail(df.shape[0] - 2000)
    df = df.reset_index(drop=True)

    # -- rescale copy to 0-1
    df2 = df.copy()
    n_min, n_max = 0, 1
    minimum, maximum = np.min(df, axis=0), np.max(df, axis=0)
    m = (n_max - n_min) / (maximum - minimum)
    b = n_min - m * minimum
    df2 = m * df2 + b

    # -- calc sum
    df['sum'] = \
    df2['Анкетный скоринг'] + \
    df2['Сумма Скорингов'] + \
    df2['Оценка кредитной истории ОКБ^'] + \
    df2['Оценка кредитной истории НБКИ^'] + \
    df2['Оценка кредитной истории Эквифакс^']
    # 'A',
    # 'B',
    # df['МБКИ']
    # 'Возраст клиента',
    # 'ander',
    # print(df['sum'].describe())
    # df = df.sort_values(by='sum')
    # -- get only required
    df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    # -- calc hist
    bins = 20
    yl, binEdges = np.histogram(df['sum'].to_numpy(), bins=bins)


    print(binEdges) # from smalles to larges`
    print(len(yl))
    print(yl)
    print(df['sum'].isna().sum())
    print(df['sum'].describe())

    # -- set sum as histogram difference appr/ rej
    import math
    appr_c_all = df[df['ander'] == 0].shape[0]
    rej_c_all = df[df['ander'] == 1].shape[0]
    for pair in zip(binEdges[:-1], binEdges[1:]):
        low, high = pair
        print("pair", pair)
        # if high > 4:
        #     return
        appr_c = df[(df['sum'] >= low) & (df['sum'] < high) & (df['ander'] == 0)].shape[0] / appr_c_all
        rej_c = df[(df['sum'] >= low) & (df['sum'] < high) & (df['ander'] == 1)].shape[0] / rej_c_all
        print(appr_c, rej_c)
        if appr_c == 0 or rej_c == 0:
            df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 0
        # elif appr_c == 0:
        #     df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 3
        # elif rej_c == 0:
        #     df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 3
        elif appr_c == rej_c:
            new_value = math.log(appr_c / rej_c)
            df.loc[(df['sum'] >= low) & (df['sum'] <= high), ['sum']] = new_value
        else:
            new_value = math.log(appr_c / rej_c)
            print("o", low, high, new_value)
            print("before", df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']])
            print(df[(df['sum'] >= low) & (df['sum'] < high)]['sum'].shape)
            df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = new_value
            print(df.loc[(df['sum'] >= new_value) & (df['sum'] <= new_value), ['sum']].shape)

    print(df.tail(10).to_string())
    print(df['sum'].describe())

    # -- get unique
    print(-last_n, last_n_slide, sorted(df['sum'].unique().tolist()))
    values = sorted(df['sum'].unique().tolist())[-last_n:last_n_slide]
    print(values)

    low = values[:1][0]
    high = values[-1:][0]
    print(low, high)

    df = df[(df['sum'] >= low) & (df['sum'] <= high)]
    # print(df.shape)
    # print(df.tail(20).to_string())
    # return

    p = 'with_sum.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def add_sum_odob_rej(p, last_n: int = 2, last_n_slide=-1):
    """ to rescale to 1 0 we requre encude all categprical columns to numbers"""
    df: pd.DataFrame = pd.read_pickle(p)
    # df['Скоринговый балл ОКБ, основной скоринг бюро'] + \
    # df['Эквифакс 4Score'] + \
    # df['Скоринговый балл НБКИ общ'] + \
    # df2['Запрошенная сумма кредита'] + \
    # df2['Сумма КИ'] + \
    df = df.tail(df.shape[0] - 2000)
    df = df.reset_index(drop=True)

    # -- rescale copy to 0-1
    df2 = df.copy()
    n_min, n_max = 0, 1
    minimum, maximum = np.min(df, axis=0), np.max(df, axis=0)
    m = (n_max - n_min) / (maximum - minimum)
    b = n_min - m * minimum
    df2 = m * df2 + b

    # -- calc sum
    df['sum'] = \
    df2['Анкетный скоринг'] + \
    df2['Сумма Скорингов'] + \
    df2['Оценка кредитной истории ОКБ^'] + \
    df2['Оценка кредитной истории НБКИ^'] + \
    df2['Оценка кредитной истории Эквифакс^']
    # 'A',
    # 'B',
    # df['МБКИ']
    # 'Возраст клиента',
    # 'ander',
    # print(df['sum'].describe())
    # df = df.sort_values(by='sum')
    # -- get only required
    df = df[(df['ander'] == 0) | (df['ander'] == 1) | (df['ander'] == 2)]
    # -- calc hist
    bins = 20
    yl, binEdges = np.histogram(df['sum'].to_numpy(), bins=bins)


    print(binEdges) # from smalles to larges`
    print(len(yl))
    print(yl)
    print(df['sum'].isna().sum())
    print(df['sum'].describe())

    # -- set sum as histogram difference appr/ rej
    import math
    appr_c_all = df[df['ander'] == 0].shape[0]
    rej_c_all = df[df['ander'] == 1].shape[0]
    for pair in zip(binEdges[:-1], binEdges[1:]):
        low, high = pair
        print("pair", pair)
        # if high > 4:
        #     return
        appr_c = df[(df['sum'] >= low) & (df['sum'] < high) & (df['ander'] == 0)].shape[0] / appr_c_all
        rej_c = df[(df['sum'] >= low) & (df['sum'] < high) & (df['ander'] == 1)].shape[0] / rej_c_all
        print(appr_c, rej_c)
        if appr_c == 0 or rej_c == 0:
            df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 0
        # elif appr_c == 0:
        #     df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 3
        # elif rej_c == 0:
        #     df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = 3
        elif appr_c == rej_c:
            new_value = math.log(appr_c / rej_c)
            df.loc[(df['sum'] >= low) & (df['sum'] <= high), ['sum']] = new_value
        else:
            new_value = math.log(appr_c / rej_c)
            print("o", low, high, new_value)
            print("before", df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']])
            print(df[(df['sum'] >= low) & (df['sum'] < high)]['sum'].shape)
            df.loc[(df['sum'] >= low) & (df['sum'] < high), ['sum']] = new_value
            print(df.loc[(df['sum'] >= new_value) & (df['sum'] <= new_value), ['sum']].shape)

    print(df.tail(10).to_string())
    print(df['sum'].describe())

    # -- get unique
    print(-last_n, last_n_slide, sorted(df['sum'].unique().tolist()))
    values = sorted(df['sum'].unique().tolist())[-last_n:last_n_slide]
    print(values)

    low = values[:1][0]
    high = values[-1:][0]
    print(low, high)

    df = df[(df['sum'] >= low) & (df['sum'] <= high)]
    # print(df.shape)
    # print(df.tail(20).to_string())
    # return

    p = 'with_sum.pickle'
    pd.to_pickle(df, p)
    print("ok")
    return p


def pereb(p:str):
    df: pd.DataFrame = pd.read_pickle(p)
    p = df

    # gn = 100
    gn = 800
    # gn = 2000
    res = {}

    for a in np.arange(0.1,1, 0.1):
        for b in np.arange(0.1, 1, 0.2):
            for c in np.arange(0.1, 1, 0.2):
                for d in np.arange(0.1, 1, 0.3):
                    for e in np.arange(0.1, 1, 0.3):
                        for f in np.arange(0.1, 1, 0.3):
                            for g in np.arange(0.1, 1, 0.4):
                                p2 = add_sum(p, gn, a,b,c,d,e,f,g)  # encode_categorical required
                                r = plot_numerical_and_categorical(p2, "Приложение 4.3 hist and bars " + str(
                                    gn))  # etap 3 # add_sum
                                res[r] = (a,b,c,d,e,f,g)
                                print(r, res[r])
        for x in sorted(res.keys())[-10:]:
            print(x, res[x])
    return max(res.keys()), res[max(res.keys())]


if __name__ == '__main__':
    from myown_pack.common import impute_v, outliers_numerical, encode_categorical, save, \
        standardization, standardization01
    from myown_pack.clusterization import hierarchical_clustering_post_anal
    p = csv_file_read('/home/u2/evseeva/Отчет по сделкам(Андеррайтер).csv')

    p = 'after_read.pickle'
    # p = select_columns(p)
    # p = 'columns_selected.pickle'
    # p = outliers_numerical(p)  # without 2000
    # p = 'without_outliers.pickle'  # Raw
    # p = impute_v(p, 11, percent=0.70, exclude=['ander'], remove=['first_decision_state', 'Коды отказа', 'Статус заявки'])  # and remove rows
    # p = 'after_imputer.pickle'
    #
    # p = encode_categorical(p)
    # p = 'encoded.pickle'
    # corr_matrix(p)
    # p = standardization(p, exclude=['ander'])  # full NA = 0
    # p = standardization01(p, exclude=['ander'])  # full NA = 0
    # p = 'standardized_rej.pickle'
    # p = 'standardized.pickle'
    # p = add_sum_forest(p, 0, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5)  # encode_categorical required
    # gn = 100
    # gn = 800
    # gn = 2000
    # p = add_sum(p, gn, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5)  # encode_categorical required
    # h, c = 6, -4
    # h, c = 4, -2
    # h, c = 2, None
    # p = add_sum_odob_rej(p, h, c)  # encide_categorical required
    # p = 'with_sum.pickle'
    # p = feature_engeering(p, exclude=['ander'], remove=['first_decision_state', 'Коды отказа', 'Статус заявки'])
    # p = feature_engeering(p, exclude=['ander'])
    # p = feature_engeering(p, exclude=['ander'], selected_columns=[
    #     'Скоринговый балл ОКБ, основной скоринг бюро',
    #     'Эквифакс 4Score',
    #     'Скоринговый балл НБКИ общ',
    #     'Анкетный скоринг',
    #     'Сумма КИ',
    #     'Сумма Скорингов',
    #     'Запрошенная сумма кредита',
    #     # 'A',
    #     # 'B'
    #                                                                  'Оценка кредитной истории ОКБ^',
    #                                                                  'Оценка кредитной истории НБКИ^',
    #                                                                  'Оценка кредитной истории Эквифакс^'
    # ])
    # p = 'feature_eng.pickle'
    # labels = hierarchical_clustering(p)
    # hierarchical_clustering_post_anal(labels, p='without_outliers.pickle')
    # p = hierarchical_clustering_filter(la123bels, p='without_outliers.pickle')
    # p = 'cluster_filter.pickle'
    from pipeline_otchet_cluster_analysis_plot import plot_boxes, plot_ekv_summa, plot_ki_scoring, plot_one_box, \
        plot_kde_plot_matrix, plot_numerical_and_categorical

    # frequency_analysis(p)
    # plot_boxes(p)  # гистограмма одна
    # plot_one_box(p)  # etap 1 without_outliers
    # plot_numerical_and_categorical(p, "Приложение 4.4 hist and bars " + str(gn))  # etap 3 # add_sum
    # plot_numerical_and_categorical(p, "Приложение 4.2 hist and bars " + str(h))  # etap 3
    # plot_kde_plot_matrix(p)
    # plot_scatter_ekv_summa(p)
    # plot_ekv_summa(p)
    # plot_ki_scoring(p)
    # pereb(p)

    # ml_clustering(p)
    # affinity_p_clustering(p)

    # forest_search_parameters(p, target='ander')
    # feature_importance_forest2(p, target='ander')
    # feature_importance_forest(p, target='ander', max_depth=8, n_estimators=2, max_leaf_nodes=7, min_samples_split=3)
    # xgboost_serch_parameters()
    # df: pd.DataFrame = pd.read_pickle(p)
    # print(df.iloc[[36141]]['Статус заявки'])
    # print(df.iloc[[2]]['Статус заявки'])
    # print(df.iloc[[3423]]['Статус заявки'])
