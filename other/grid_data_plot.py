import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
# make up some randomly distributed data


def griddata_plot(x, y, title, x_l, y_l, cmap):
    from scipy.interpolate import griddata
    lims1 = [np.min(np.abs(x)), np.max(np.abs(x))]
    bins1 = np.linspace(lims1[0], lims1[1], 20)
    lims2 = [np.min(np.abs(y)), np.max(np.abs(y))]
    bins2 = np.linspace(lims2[0], lims2[1], 20)

    normed = False
    weights = None
    range = None
    h, xe, ye = np.histogram2d(x, y, bins=(bins1, bins2),
                                   range=range, normed=normed,
                                   weights=weights)
    from hist_scatter import map_hist
    dens = map_hist(x, y, h, bins=(xe, ye))
    z = np.nan_to_num(dens)
    # define grid.
    # xi = np.linspace(lims2[0], lims1[1], 20)
    # yi = np.linspace(lims2[0], lims2[1], 20)
    xi = bins1
    yi = bins2
    # grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')  #'linear'
    # Plot contour the gridded data, plotting dots at the randomly spaced data points.
    fig = plt.figure(figsize=(15,8))
    CS = plt.contour(xi,yi,zi,25,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,15,cmap=cmap)  # plt.cm.cool# plt.cm.gist_rainbow
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='o', c='b', s=0.2)
    # plt.xlim(-2,2)
    # plt.ylim(-2,2)
    plt.title(title)
    ax = plt.gca()
    ax.set_xlabel(x_l)
    ax.set_ylabel(y_l)
    # ax.set_xlim(left=500000, right=2600000)
    # ax.set_ylim(bottom=500)
    plt.savefig(title)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    seed(13)

    rng = np.random.RandomState(10)
    x = np.hstack((rng.normal(loc=10, scale=2, size=1000), rng.normal(loc=15, scale=2, size=1000)))
    # print(x)
    y = np.hstack((rng.normal(loc=10, scale=2, size=1000), rng.normal(loc=15, scale=2, size=1000)))
    griddata_plot(x,y)