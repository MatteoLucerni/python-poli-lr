import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep=r'\s+', names=["CRIM", "ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])

cols=['RM', 'LSTAT', 'DIS', 'RAD', 'MEDV']

sns.pairplot(boston[cols])
plt.show()

X = boston[['LSTAT']].values
Y = boston[['MEDV']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

for i in range(1,11):
    polyfeats = PolynomialFeatures(degree=i)

    X_train_poly = polyfeats.fit_transform(X_train)

    X_test_poly = polyfeats.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train)
    Y_pred = lr.predict(X_test_poly)

    mse = mean_squared_error(Y_test, Y_pred)
    r2s = r2_score(Y_test, Y_pred)

    print('Deg: ' + str(i) + ' / Error: ' + str(mse) + ' / Score: ' + str(r2s))

print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')

# all colums
X = boston.drop('MEDV', axis=1).values
Y = boston[['MEDV']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# overfitting i > 2
for i in range(1,5):
    polyfeats = PolynomialFeatures(degree=i)

    X_train_poly = polyfeats.fit_transform(X_train)

    X_test_poly = polyfeats.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train)
    Y_pred = lr.predict(X_test_poly)

    mse = mean_squared_error(Y_test, Y_pred)
    r2s = r2_score(Y_test, Y_pred)

    print('Deg: ' + str(i) + ' / Error: ' + str(mse) + ' / Score: ' + str(r2s))