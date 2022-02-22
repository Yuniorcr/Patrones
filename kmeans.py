import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('data/candy.csv')
    X = df.drop('competitorname', axis=1)

    kmeans = KMeans(n_clusters=4)
    miniBatchKMeans = MiniBatchKMeans(n_clusters=4, batch_size=8)

    kmeans.fit(X)
    miniBatchKMeans.fit(X)

    print(kmeans.predict(X))
    print(miniBatchKMeans.predict(X))

    df['kmeans'] = kmeans.predict(X)	# Add a column to the dataframe
    print(df.head())
    
    plt.scatter(df['sugarpercent'], df['pricepercent'], c=df['kmeans'], s=50, cmap='viridis')
    plt.show()