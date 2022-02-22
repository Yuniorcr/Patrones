import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if  __name__=='__main__':
    # Load the data
    data = pd.read_csv('./data/whr2017.csv')
    # print(data.head())

    X = data[['gdp', 'family', 'lifexp', 'freedom', 'generosity', 'corruption', 'dystopia']]
    y = data['score']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the linear regression model
    lr = LinearRegression().fit(X_train, y_train)
    y_predict_lr = lr.predict(X_test)

    # Create the Lasso model
    lasso = Lasso(alpha=0.2).fit(X_train, y_train)
    y_predict_lasso = lasso.predict(X_test)

    # Create the Ridge model
    ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = ridge.predict(X_test)

    # Calculate the MSE
    mse_lr = mean_squared_error(y_test, y_predict_lr)
    mse_lasso = mean_squared_error(y_test, y_predict_lasso)
    mse_ridge = mean_squared_error(y_test, y_predict_ridge)

    print('Linear Regression MSE: ', mse_lr)
    print('Lasso Regression MSE: ', mse_lasso)
    print('Ridge Regression MSE: ', mse_ridge)

    # coefficients
    print('Linear Regression Coefficients: ', lr.coef_)
    print('Lasso Regression Coefficients: ', lasso.coef_)
    print('Ridge Regression Coefficients: ', ridge.coef_)
