import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import skewtest, skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans

from sklearn.mixture import GaussianMixture
from sklearn.utils import resample

from sklearn import metrics


def standartize_for_clustering(df: pd.DataFrame, median_mean=None, ignore=None):
    """ way too old, not used - replaced with step: encoding one-hot"""

    def _scale_clusterization(df, c):
        # -- scale for clusterization
        d = df[c].to_numpy().reshape(-1, 1)
        # print("a", c, skew(df[c]) > 1, len(df[c].unique()))
        if median_mean:
            if c in median_mean:
                if median_mean[c] == 'median':
                    df[c] = (df[c] - np.median(df[c])) / df[c].std()
                else:
                    df[c] = (df[c] - np.mean(df[c])) / df[c].std()
                return df

        if abs(skew(df[c])) > 1:  # and len(df[c].unique()) < 5
            # df[c] = MinMaxScaler((-1, 1)).fit_transform(d)  # increase importance by 2
            # df[c] = (df[c] - np.median(df[c]))
            print("medi", c, skew(df[c]), "uniq", len(df[c].unique()), "median", np.median(df[c]), "mean",
                  np.mean(df[c]))
            df[c] = (df[c] - np.median(df[c])) / df[c].std()
        else:
            print("mean", c, skew(df[c]), "uniq", len(df[c].unique()), "median", np.median(df[c]), "mean",
                  np.mean(df[c]))
            df[c] = (df[c] - np.mean(df[c])) / df[c].std()
            # df[c] = StandardScaler().fit_transform(d)
        return df

    # df = df.sample(n_samples, random_state=2, replace=False)
    # X: np.array = resample(X, n_samples=n_samples, random_state=42)
    # df = df.reset_index(drop=True)


    # df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    # print(df.isna().sum())
    for c in df.columns.tolist():
        if c in ignore:
            continue
        _scale_clusterization(df, c)

    return df


def adjust_for_one_hot_columns(df:pd.DataFrame, oh_cols):
    cols = df.columns.tolist()
    for tc in oh_cols:
        sel_cols = [c for c in cols if c.startswith(tc)]
        for c in sel_cols:
            s_count = df[c].value_counts()
            print(df[c].isna().sum())
            # important values count > size/15
            important_unique_values = s_count[s_count > df.shape[0] / 15].shape[0]
            print("Important unique values:")
            print(c, important_unique_values)
            print()
        for c in sel_cols:
            # print(c, len(sel_cols))
            df[c] = df[c] / len(sel_cols)
    return df


def cluster_loop_pca(p, n_clusters):
    """
    required standartization
    :param p:
    :return:
    """
    from sklearn.decomposition import KernelPCA
    from sklearn import metrics
    df: pd.DataFrame = pd.read_pickle(p)
    df = df.drop(columns=['id'])
    # linkage = 'single'
    linkage = 'average'
    # linkage='complete'
    # linkage = 'ward'
    #
    # affinity = "manhattan"
    # affinity='manhattan'
    affinity = 'euclidean'
    # affinity = 'cosine'
    # affinity='euclidean'

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    df.reset_index(inplace=True, drop=True)
    # weight for feature
    ander = df['ander'].to_numpy().copy()
    # test_c = 'EQUIFAX_RATING_SCORING_КИ отсутствует'
    test_c = 'CLIENT_AGE'
    test = df[test_c].to_numpy().copy()
    pca = KernelPCA(n_components=2,
                    kernel='rbf',
                    # kernel='linear',
                    # kernel='sigmoid',
                    # kernel='cosine',
                    )
    # l = [0, 0.5,0.6, 0.7,0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 1, 1.05, 1.1, 1.2, 1.3]
    # l = [1, 2,3,4,5]
    l = [0.001, 0.5, 1, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3, 4]
    for i, n in enumerate(l):
        df['ander'] = ander * n
        # df[test_c] = test * n
        # print(n)

        X_principal = pca.fit_transform(df)
        X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])
        # model = AgglomerativeClustering(n_clusters=20, affinity=affinity, linkage=linkage) #
        model = GaussianMixture(n_components=n_clusters, random_state=2)
        # model = KMeans(n_clusters=n_clusters, random_state=2)
        # model = SpectralClustering(n_clusters=n_clusters, random_state=2)
        # model = MiniBatchKMeans(n_clusters=2)
        # model = DBSCAN(eps=2, min_samples=150
        labels = model.fit_predict(df)
        # -- plot
        plt.figure(figsize=(10, 10))
        # plt.scatter(X_principal['P1'], X_principal['P2'], c=model.fit_predict(X_principal), cmap='rainbow', s=20)
        plt.scatter(X_principal['P1'], X_principal['P2'], c=labels, cmap='rainbow', s=20)
        t = f"Условная форма кластеров построенная по уменьшенной размерности {n}.png"
        plt.title(t + f"\n calinski_harabasz_score {metrics.calinski_harabasz_score(df, labels)}" +
                  # f"\n mutual_info_score {metrics.adjusted_mutual_info_score(ander, labels)}" +
                  f"\n contingency_matrix {metrics.cluster.contingency_matrix(ander, labels)}")
        sf = f'{int(i/10)}{i%10}pca.png'
        # plt.legend([0], )
        plt.savefig(sf)
        plt.close()
        # -- print
        print(i, n, sf)
        print(f"rand_score {metrics.adjusted_rand_score(ander, labels)}")
        print(f"mutual_info_score {metrics.adjusted_mutual_info_score(ander, labels)}")
        # print(f"silhouette_score {metrics.silhouette_score(df, labels)}")   # bad
        print(f"calinski_harabasz_score {metrics.calinski_harabasz_score(df, labels)}")
        print(f"davies_bouldin_score {metrics.davies_bouldin_score(df, labels)}")

        print(f"contingency_matrix {metrics.cluster.contingency_matrix(ander, labels)}")


def pca_for_labels(p, labels, i=1):
    """
    required prepare_for_clustering or standartization
    :param p:
    :return:
    """
    from sklearn.decomposition import KernelPCA
    df: pd.DataFrame = pd.read_pickle(p)

    from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    df.reset_index(inplace=True, drop=True)
    # weight for feature
    ander = df['ander'].to_numpy().copy()
    pca = KernelPCA(
        # kernel='rbf',
        kernel='linear',
        # kernel='sigmoid',
        # kernel='cosine',
        n_components=2)
    df['ander'] = ander * i

    X_principal = pca.fit_transform(df)
    X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])

    plt.figure(figsize=(10, 10))
    print(labels, len(labels), X_principal.shape)
    print(np.unique(labels))
    scatter = plt.scatter(X_principal['P1'], X_principal['P2'], c=labels, cmap='rainbow', s=20)
    t = f"Условная форма кластеров построенная по уменьшенной размерности{i}.png"
    plt.title(t)
    cl = [str(x) for x in np.unique(labels)]
    plt.legend(handles=scatter.legend_elements()[0], labels=cl)
    plt.show()
    plt.savefig(f'{i}pca.png')
    plt.close()


def pca_clustering(p, i=1, n_clusters=2):
    """
    required prepare_for_clustering or standartization

    На PCA или не на PCA легко переключить, отправив алгоритму PCA principals или df

    :param p:
    :return:
    """
    from sklearn.decomposition import KernelPCA
    df_o: pd.DataFrame = pd.read_pickle(p)
    df_o = df_o.drop(columns=['id'])

    df_o.reset_index(inplace=True, drop=True)
    # weight for feature
    ander = df_o['ander'].to_numpy().copy()
    df_o = df_o[df_o['ander'] == 1]
    # df_o['ander'] = ander * i
    # df['EQUIFAX_RATING_SCORING_Хорошая КИ'] = df['EQUIFAX_RATING_SCORING_Хорошая КИ'] *0.00001
    # df['EQUIFAX_RATING_SCORING_КИ отсутствует'] = df['EQUIFAX_RATING_SCORING_КИ отсутствует'] * 1.2
    df_o['CLIENT_AGE'] = df_o['CLIENT_AGE'] * 0.0001
    # df['ANKETA_SCORING'] = df['ANKETA_SCORING'] * 0.0001
    df_o['AUTO_DEAL_INITIAL_FEE'] = df_o['AUTO_DEAL_INITIAL_FEE'] * 0.00001
    # df['EQUIFAX_SCORING'] = df['EQUIFAX_SCORING'] * 0.00001
    df_o['NBKI_SCORING'] = df_o['NBKI_SCORING'] * 1.2
    df = df_o.copy()

    # -- PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    pca = KernelPCA(
        kernel='rbf',
        # kernel='linear',
        # kernel='sigmoid',
        # kernel='cosine',
        n_components=3)

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

    ac = AgglomerativeClustering(  # ,
        linkage=linkage,
        # affinity=affinity,
        n_clusters=n_clusters,
        compute_full_tree=False,
        compute_distances=True)
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import OPTICS
    from sklearn.cluster import Birch
    # ac = AffinityPropagation(damping=0.5) #, random_state=0, max_iter=50, verbose=True)
    # ac = OPTICS()
    # ac = Birch(n_clusters=3)
    # from sklearn.cluster import KMeans
    # ac = KMeans(n_clusters=4)
    from sklearn.mixture import GaussianMixture
    # ac = GaussianMixture(n_components=n_clusters, random_state=3)
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

    # df[df[nc] == 5] = 4

    print(df[nc].tolist())

    labels = df[nc].tolist()

    # -- merge
    # print(f"rand_score {metrics.adjusted_rand_score(ander, labels)}")
    # print(f"mutual_info_score {metrics.adjusted_mutual_info_score(ander, labels)}")
    # print(f"silhouette_score {metrics.silhouette_score(df, labels)}")   # bad
    print(f"calinski_harabasz_score {metrics.calinski_harabasz_score(df, labels)}")
    # print(f"davies_bouldin_score {metrics.davies_bouldin_score(df, labels)}")
    #
    # print(f"contingency_matrix {metrics.cluster.contingency_matrix(ander, labels)}")
    # labels = ac.fit_predict(df)
    # -- plot
    pca = KernelPCA(
        kernel='rbf',
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


def hierarchical_clustering_plot(hc_model, dendr_level=None):
    print("clusters count:", hc_model.n_clusters_)  # количество кластеров по trashold

    # -- create the counts of samples under each node
    counts = np.zeros(hc_model.children_.shape[0])
    n_samples = len(hc_model.labels_)
    for i, merge in enumerate(hc_model.children_):
        # i - cluster index at i iteration
        # counts[i] - count of rows
        # counts of child:
        # child_idx = children_[i][0/1] if < n_samples - leaf=1, else inner node = int(counts[child_idx - n_samples])
        # print(merge)
        current_count = 0  # count of rows in new cluster
        for child_idx in merge:  # (two)
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += int(counts[child_idx - n_samples])  # (child_idx - n_samples) - step i
        counts[i] = current_count
        # -- print merge log
        i1 = 'leaf__' + str(merge[0]) if merge[0] < n_samples else "id_" + str(merge[0] - n_samples)
        i2 = 'leaf__' + str(merge[1]) if merge[1] < n_samples else "id_" + str(merge[1] - n_samples)
        print(i, i1, '\t', i2, '\t', current_count, 'd_' + str(round(hc_model.distances_[i], 2)))

    # -- Sreen Plot
    plt.scatter(range(len(hc_model.distances_), 0, -1), hc_model.distances_, s=3)  # step, distance
    plt.gca().set(xlabel='кластеры', ylabel='расстояния между кластерами')
    plt.show()
    # --dendra

    from scipy.cluster.hierarchy import dendrogram
    linkage_matrix = np.column_stack([hc_model.children_, hc_model.distances_,
                                      counts]).astype(float)
    if dendr_level is None:
        dendr_level = input("какой уровень использовать для Дендрограммы?\n")
        dendr_level = int(dendr_level)
    dendrogram(linkage_matrix, truncate_mode='level', p=dendr_level)
    plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.xlabel("Количество точек (или номер заявки если со скобками).")
    plt.show()
    plt.savefig('d')
    plt.close()
    print(len(hc_model.labels_), hc_model.labels_)


def hierarchical_clustering(p):
    from sklearn.cluster import AgglomerativeClustering

    df: pd.DataFrame = pd.read_pickle(p)
    # replace na
    df.fillna(0, inplace=True)
    # df = df.head(2000)
    print(df.shape)
    # return

    X: np.array = df.to_numpy()
    # distance_threshold - влияет на n_clusters_ и labels_
    hc_model = AgglomerativeClustering(n_clusters=None, affinity='euclidean',
                                       # affinity='manhattan', #  # affinity='manhattan',
                                       compute_full_tree=True, linkage='single',
                                       # linkage='complete', #   # linkage='average', # linkage='single', linkage='ward'
                                       compute_distances=True, distance_threshold=100000)  # 186  distance_threshold=125
    hc_model.fit(X)
    print("clusters count:", hc_model.n_clusters_)  # количество кластеров по trashold

    # -- create the counts of samples under each node
    counts = np.zeros(hc_model.children_.shape[0])
    n_samples = len(hc_model.labels_)
    for i, merge in enumerate(hc_model.children_):
        # i - cluster index at i iteration
        # counts[i] - count of rows
        # counts of child:
        # child_idx = children_[i][0/1] if < n_samples - leaf=1, else inner node = int(counts[child_idx - n_samples])
        # print(merge)
        current_count = 0
        for child_idx in merge:  # (two)
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += int(counts[child_idx - n_samples])  # (child_idx - n_samples) - step i
        counts[i] = current_count
        # -- print merge log
        i1 = '__leaf__' if merge[0] < n_samples else "id_" + str(merge[0] - n_samples)
        i2 = '__leaf__' if merge[1] < n_samples else "id_" + str(merge[1] - n_samples)
        # print(i, i1, '\t', i2, '\t', current_count, 'd_' + str(int(hc_model.distances_[i])))

    # -- Sree Plot

    plt.scatter(range(len(hc_model.distances_), 0, -1), hc_model.distances_)  # step, distance
    plt.gca().set(xlabel='кластеры', ylabel='расстояния между кластерами')
    plt.show()
    # --dendra

    from scipy.cluster.hierarchy import dendrogram
    linkage_matrix = np.column_stack([hc_model.children_, hc_model.distances_,
                                      counts]).astype(float)
    dendr_level = input("какой уровень использовать для Дендрограммы?\n")
    dendr_level = int(dendr_level)
    dendrogram(linkage_matrix, truncate_mode='level', p=dendr_level)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    print(len(hc_model.labels_), hc_model.labels_)
    return hc_model.labels_


def hierarchical_clustering_post_anal(labels: list, p, columns: list):
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

    from matplotlib import pyplot as plt
    for c in columns:
        res = list()
        for i in set(labels):
            res.append(dfs[i][c])
    plt.boxplot(res)
    plt.title(c)
    plt.show()


def ml_clustering_em(p):
    df: pd.DataFrame = pd.read_pickle(p)
    from sklearn.mixture import GaussianMixture
    # print(df.shape)
    # df = df.head(2000)
    # -- one
    gmm = GaussianMixture(3, covariance_type='full', random_state=0)
    gmm.fit(df)
    # labels = gmm.predict(df)
    # print(labels)
    # -- several
    n_m = range(1, 20)
    models = [GaussianMixture(n, covariance_type='full', random_state=3).fit(df) for n in n_m]
    gmm_model_comparision = pd.DataFrame({"n_clasters": n_m,
                                          "BIC": [m.bic(df) for m in models],
                                          "AIC": [m.aic(df) for m in models]})
    print(gmm_model_comparision)
    from matplotlib import pyplot as plt
    plt.plot(list(n_m), gmm_model_comparision['AIC'], label='AIC')
    plt.plot(list(n_m), gmm_model_comparision['BIC'], label='BIC')
    plt.legend()
    plt.gca().set(xlabel='число кластеров', ylabel='оценка модели')
    plt.show()


def affinity_p_clustering(p):
    df: pd.DataFrame = pd.read_pickle(p)
    from sklearn.cluster import AffinityPropagation
    import time
    t1 = time.time()
    m = AffinityPropagation(damping=0.5, random_state=0, max_iter=400, verbose=True)
    m.fit(df)
    n_clusters = len(m.cluster_centers_indices_)
    print(n_clusters)
    print("time min ", round((time.time() - t1) / 60))


def hierarchical_clustering_with_custom_metric(p):
    from sklearn.metrics.pairwise import pairwise_distances
    # https://gist.github.com/codehacken/8b9316e025beeabb082dda4d0654a6fa
    df: pd.DataFrame = pd.read_pickle(p)
    # columns
    cols = df.columns.tolist()
    cc = []
    for c in cols:
        if len(df[c].unique()) == 2:
            cc.append(c)
    print(cc)

    def metric_my(x, y):
        """
        global cols
        :param x: one row
        :param y: one row
        :return: distance
        """
        distance = 0
        for c in cols:
            # print(c)
            i = cols.index(c)
            if c == 'ander':
                s = x[i] + y[i]
                if s == -2 or s == 2:
                    distance += 0
                else:
                    distance += 1  # 8
            # -- binary
            if c in cc:
                distance += abs(x[i] - y[i])  # 1 max
                # print("wtf", distance)
        # print(distance)

        return distance

    distance_matrix = pairwise_distances(df, df, metric=metric_my)  # (columns_opunt, columns_opunt)
    print(distance_matrix)
    hc_model = AgglomerativeClustering(n_clusters=None,
                                       affinity='precomputed',
                                       compute_full_tree=True,
                                       linkage='single',
                                       compute_distances=True, distance_threshold=200010)  # 186  distance_threshold=125
    hc_model.fit_predict(distance_matrix)


def hierarchical_clustering(p, distance_threshold=0, i=1):
    df: pd.DataFrame = pd.read_pickle(p)

    df.reset_index(inplace=True, drop=True)
    # print(df.columns)

    # linkage = 'single'
    linkage = 'average'
    # linkage='complete'
    # linkage = 'ward'
    #
    # affinity = "manhattan"
    affinity = 'manhattan'
    # affinity = 'euclidean'
    # affinity='euclidean'

    df.reset_index(drop=True, inplace=True)
    i = i
    df['ander'] = df['ander'].to_numpy() * i
    df.reset_index(drop=True, inplace=True)

    ac2 = AgglomerativeClustering(  # ,
        linkage=linkage,
        affinity=affinity,
        n_clusters=2,
        compute_full_tree=False,
        compute_distances=True)

    from sklearn.decomposition import KernelPCA
    pca = KernelPCA(n_components=2)
    X_principal = pca.fit_transform(df)
    X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])

    # cluster_labels = ac2.fit_predict(X_principal)



    # model = AgglomerativeClustering(
    #                                    n_clusters=None,
    #                                    affinity=affinity,
    #                                    compute_full_tree=True,
    #                                    linkage=linkage,
    #                                    compute_distances=True,
    #                                    distance_threshold=distance_threshold
    #                                    )  # 186  distance_threshold=125

    from sklearn.mixture import GaussianMixture
    model = GaussianMixture(n_components=4)

    labels = model.fit_predict(df)

    plt.figure(figsize=(10, 10))
    plt.scatter(X_principal['P1'], X_principal['P2'], c=labels, cmap='rainbow', s=20)
    plt.title(f"Условная форма кластеров построенная по уменьшенной размерности ({i})")
    plt.savefig(f"Условная форма кластеров построенная по уменьшенной размерности ({i}).png")
    plt.show()

    return model, labels


if __name__ == '__main__':
    pass
