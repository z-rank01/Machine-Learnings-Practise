import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolutions=0.02):
    """Plot the decision result
    
    Parameters
    ----------
    X: training data matrix
    y: labels vector
    classifier: target classifier
    resolutons: region boundary resolutions

    Return
    ------
    Nothing, plot the result

    """
    # set up colors and markers map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    markers= ('s', 'x', 'o', '^', 'v')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # we use ListedColormap to find subset of colors whose number meets the labels

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolutions), 
                           np.arange(x2_min, x2_max, resolutions))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples
    for idx, col in enumerate(np.unique(y)):
        plt.scatter(X[y==col, 0], 
                    X[y==col, 1], 
                    alpha=0.6, 
                    color=cmap(idx), 
                    edgecolors='black', 
                    marker=markers[idx], 
                    label=col)


