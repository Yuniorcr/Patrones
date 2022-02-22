import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == '__main__':
    df = pd.read_csv('data/candy.csv')
    X = df.drop('competitorname', axis=1)

    ms = MeanShift()
    ms.fit(X)

    print(ms.labels_)
    print(ms.cluster_centers_)

    df['ms'] = ms.labels_
    print(df)
