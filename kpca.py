import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA # Kernel PCA (Kernel Principal Component Analysis)

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

    kpca = KernelPCA(n_components=3, kernel="poly") # kernel = "rbf", "sigmoid", "poly"
    kpca.fit(X_train)

    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)

    LR = LogisticRegression( solver="lbfgs")

    LR = LR.fit(X_train_kpca, y_train)
    print("KPCA: ", LR.score(X_test_kpca, y_test))
