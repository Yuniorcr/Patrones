import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('./data/heart.csv')

    # print(df['target'].describe())
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classifier = {
        'KNN': KNeighborsClassifier(),
        'LinearSVC': LinearSVC(),
        'SVC': SVC(),
        'SGD': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for train_name, train_classifier in classifier.items():
        train_classifier.fit(X_train, y_train)
        y_pred = train_classifier.predict(X_test)
        print(train_name, accuracy_score(y_pred, y_test ))
        Bagging_classifier = BaggingClassifier(train_classifier, n_estimators=10)
        Bagging_classifier.fit(X_train, y_train)
        y_pred_bagging = Bagging_classifier.predict(X_test)
        print('{} accuracy: {}'.format(train_name, accuracy_score(y_pred_bagging, y_test )))
        