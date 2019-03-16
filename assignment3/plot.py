import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_scatter_boundary(model, X_set, y_set, title, step_size=0.1, xlabel='x1', ylabel='x2'):
    '''Plot the decision boundary of the model and a scatter plot of the data points'''
    X_grid_0 = np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=step_size)
    X_grid_1 = np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=step_size)
    X1 , X2 = np.meshgrid(X_grid_0, X_grid_1)

    X3 = model.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)

    plt.contourf(X1, X2, X3, alpha=0.50, cmap=ListedColormap(("yellow", "cyan")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            label = j,
            edgecolors = "Black"
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_clusters(cluster_model, X_set, title, xlabel='x1', ylabel='x2'):
    x1, x2 = X_set[:, 0], X_set[:, 1]
    plt.scatter(x1, x2, c=cluster_model.assignments, zorder=1)

    plt.scatter(cluster_model.centroids_states[0, :, 0], cluster_model.centroids_states[0, :, 1], c='black', s=200, marker='X', zorder=3)
    for i in range(cluster_model.K):
        c_states = cluster_model.centroids_states[:, i]  # get all states of centroid i
        plt.plot(c_states[:, 0], c_states[:, 1], c='cyan', linestyle='dashed', marker='.', zorder=2)
    plt.scatter(cluster_model.centroids[:, 0], cluster_model.centroids[:, 1], c='red', s=200, marker='X', zorder=3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()