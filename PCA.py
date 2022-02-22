import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # Principal Component Analysis
from sklearn.decomposition import IncrementalPCA # Incremental PCA

from sklearn.linear_model  import LogisticRegression # Logistic Regression
from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("./data/heart.csv")

    # Drop column target
    X = df.drop(["target"], axis=1)
    y = df["target"]
    
    X = StandardScaler().fit_transform(X) # Standardize data (0 mean, 1 stdev)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(X_train.shape, y_train.shape)

    #n_components = min(n_muestras, n_features) 
    рса = PCA(n_components=3) 
    рса.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10) # batch_size=10 is block size
    ipca.fit(X_train)

    plt.plot(range(len(рса.explained_variance_)), рса.explained_variance_ratio_, label="PCA")
    # plt.show()
    
    LR = LogisticRegression( solver="lbfgs")

    X_train_pca = рса.transform(X_train)
    X_test_pca = рса.transform(X_test)
    LR = LR.fit(X_train_pca, y_train)
    print("PCA: ", LR.score(X_test_pca, y_test))

    X_train_ipca = ipca.transform(X_train)
    X_test_ipca = ipca.transform(X_test)
    LR = LR.fit(X_train_ipca, y_train)
    print("IPCA: ", LR.score(X_test_ipca, y_test))
    