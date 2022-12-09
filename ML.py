import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('Student-Performance\student-mat.csv', sep = ';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
print('Coefficient: \n',linear.coef_)
print('Intercept: \n',linear.intercept_)