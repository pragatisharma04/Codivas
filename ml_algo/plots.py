import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sns.set_style("darkgrid")

def plotData(outlier_values,X):
    sc=StandardScaler()
    X_scaled = sc.fit_transform(X)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1])
    plt.title("Isolation Forest - All data",
            fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("Isolation Forest Detection.png", dpi=80)
    plt.show()


    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1])
    sns.scatterplot(x=outlier_values_pca[:,0],
                    y=outlier_values_pca[:,1], color='r' )
    plt.title("Isolation Forest - anomalies in red",
            fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("Isolation Forest Detection.png", dpi=80)
    plt.show()