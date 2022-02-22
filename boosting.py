import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    
    # classifier = {
    #     'LinearSVC': LinearSVC(),
    #     'SVC': SVC(),
    #     'SGD': SGDClassifier(),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'boost': GradientBoostingClassifier(n_estimators=50)
    # }

    # for train_name, train_classifier in classifier.items():
    #     train_classifier.fit(X_train, y_train)
    #     boost_pred = train_classifier.predict(X_test)
    #     print(train_name, accuracy_score(boost_pred, y_test))

    
    for i in range(1, 201):
        boost = GradientBoostingClassifier(n_estimators=i)
        boost.fit(X_train, y_train)
        y_pred_boost = boost.predict(X_test)
        print('{} accuracy: {}'.format(i, accuracy_score(y_pred_boost, y_test)))