import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv", sep=";")
# Since our data is seperated by semicolons we need to do sep=";"

data.shape

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=.2)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(x_train,y_train)
acc = reg.score(x_test,y_test)
print(acc)

print("Coefficient : ",reg.coef_)
print("Intercept : ",reg.intercept_)

y_pred=reg.predict(x_test)

error=mean_squared_error(y_test,y_pred)
print(error)

# Drawing and plotting model
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()