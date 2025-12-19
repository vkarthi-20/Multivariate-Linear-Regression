# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step 1: Import the required libraries:

NumPy for numerical operations

Matplotlib for plotting graphs

scikit-learn modules for dataset loading, model creation, and data splitting

Step 2: Load the California Housing dataset using fetch_california_housing().

Step 3: Define:

Feature matrix X using housing.data

Target (response) vector y using housing.target

Step 4: Split the dataset into training and testing sets using train_test_split() with:

60% training data

40% testing data

Fixed random state for reproducibility

Step 5: Create a Linear Regression model object using LinearRegression().

Step 6: Train the model by fitting it to the training data (X_train, y_train) using the fit() method.

Step 7: Obtain and display the regression coefficients of the trained model.

Step 8: Evaluate the model performance by calculating the variance score (R² score) on the test dataset using the score() method.

Step 9: Set the plot style for visualization.

Step 10: Compute and plot the residual errors for:

Training data

Testing data

Residual error = Predicted value − Actual value

Step 11: Draw a horizontal reference line at zero to visualize error distribution.

Step 12: Add legend and title to the graph.

Step 13: Display the residual error plot.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# load the california housing dataset
housing = fetch_california_housing()

# defining feature matrix(X) and response vector(y)
X = housing.data
y = housing.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model
reg.fit(X_train, y_train)

# regression coefficients
print("Coefficients :", reg.coef_)

# variance score
print("Variance score:", reg.score(X_test, y_test))

# plot style
plt.style.use('fivethirtyeight')

# residual errors (train)
plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# residual errors (test)
plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# zero line
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()





```
Developed by : Karthi V
Register no: 25017522
## Output:
![WhatsApp Image 2025-12-19 at 10 15 59 PM](https://github.com/user-attachments/assets/feae8a77-3004-49bd-8a04-b3125aa5195e)
![WhatsApp Image 2025-12-19 at 10 16 11 PM](https://github.com/user-attachments/assets/a5b79221-f974-4d47-a1dd-4796ba47ac8f)



## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
