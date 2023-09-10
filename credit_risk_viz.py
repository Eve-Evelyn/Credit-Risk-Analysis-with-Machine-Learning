from credit_risk_ml import X_train_scaled, X_test_scaled, y_train, y_test, logreg
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# extract the  coefficients and intercept from the logistic regression
w_1_lr, w_2_lr = logreg.coef_[0]
w_0_lr = logreg.intercept_[0]
# use the coefficients and intercept to calculate the slope and constant for the decision boundary line
m_lr = -w_1_lr / w_2_lr
c_lr = -w_0_lr / w_2_lr

# combine X and y into the same dataframe for train and test data
data_train_scaled = X_train_scaled.copy()
data_train_scaled["Default"] = y_train
data_test_scaled = X_test_scaled.copy()
data_test_scaled["Default"] = y_test

# construct scatterplot for train and test data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

ax1 = sns.scatterplot(x="Amount",
                      y="Rate",
                      hue="Default",
                      data=data_train_scaled,
                      ax=ax[0])
ax2 = sns.scatterplot(x="Amount",
                      y="Rate",
                      hue="Default",
                      data=data_test_scaled,
                      ax=ax[1])

# add the decision boundary line to the scatter plot
x_support = np.linspace(data_test_scaled["Amount"].min(),
                        data_test_scaled["Amount"].max(), 101)
y_support = m_lr * x_support + c_lr

ax1.plot(x_support, y_support, color='black', linestyle='dashed', linewidth=3)
ax2.plot(x_support, y_support, color='black', linestyle='dashed', linewidth=3)

# set title and axis label
ax1.set_xlabel("Loan Amount", fontsize=12)
ax1.set_ylabel("Interest Rate", fontsize=12)
ax1.set_title("Train Data", fontsize=14)

ax2.set_xlabel("Loan Amount", fontsize=12)
ax2.set_ylabel("Interest Rate", fontsize=12)
ax2.set_title("Test Data", fontsize=14)

plt.show()
