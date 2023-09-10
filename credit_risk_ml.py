import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv('credit_risk.csv')

# drop all null data
data = data.dropna()
# print(data.isna().sum())

# print(data["Default"].value_counts())
# found that the default and not default data is unbalance with less default data
# use undersampling method to balance out the data so default:not default is 50:50
data_def = data[data["Default"] == "Y"]
data_non_def = data[data["Default"] == "N"]
data_non_def = data_non_def.sample(n=len(data_def), random_state=123)
data = pd.concat([data_def, data_non_def])
# print(data["Default"].value_counts())

# define the input and output
X = data[["Amount", "Rate"]]
y = data["Default"]
# convert Y/N in default to 1/0
y = y.apply(lambda x: 1 if x == "Y" else 0)
# split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=123)


# transform the X data with standard scaling as loan rate and amount is drastically different in scale
def scaler_transform(data, scaler=StandardScaler()):
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = data.columns
    data_scaled.index = data.index
    return data_scaled


X_train_scaled = scaler_transform(data=X_train)

# use gridsearch to find optimal solver and C
logreg = LogisticRegression(random_state=123)
parameters = {'solver': ['liblinear', 'saga'],
              'C': np.logspace(0, 10, 10)}
logreg_cv = GridSearchCV(estimator=logreg,
                         param_grid=parameters,
                         cv=5)
logreg_cv.fit(X=X_train_scaled, y=y_train)
# print(logreg_cv.best_params_)

# logistic regression using the best parameters
logreg = LogisticRegression(C=logreg_cv.best_params_['C'],
                            solver=logreg_cv.best_params_['solver'],
                            random_state=123)
# fit the model to train data
logreg.fit(X_train_scaled, y_train)
y_pred_train = logreg.predict(X_train_scaled)
# check the confusion matrix and classification report for train data
print(confusion_matrix(y_true=y_train, y_pred=y_pred_train))
print(classification_report(y_true=y_train,
                            y_pred=y_pred_train,
                            target_names=["Not default", "default"]))

# apply the logistic regression model to the test data
X_test_scaled = scaler_transform(data=X_test)
y_pred_test = logreg.predict(X_test_scaled)
# check the confusion matrix and classification report for test data
print(confusion_matrix(y_true=y_test, y_pred=y_pred_test))
print(classification_report(y_true=y_test,
                            y_pred=y_pred_test,
                            target_names=["Not default", "default"]))
