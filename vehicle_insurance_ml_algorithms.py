# A lot of methods and ideas where taking from these kaggle respositories
# LINK: https://www.kaggle.com/code/ayushs9020/health-insurance-prediction-eda
# LINK: https://www.kaggle.com/code/zeus75/insurance-cross-sell-prediction#Prepare-Workspace

# **DATASET PRE-PROCESSING**

import numpy as np 
import pandas as pd 

# **DATA VISUALIZATION**

import seaborn as sns 
import matplotlib.pyplot as plt

# **DATA PREPROCESSING**

from sklearn.preprocessing import FunctionTransformer , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# **MACHINE LEARNING MODELS**

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# **METRICS**

from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score

# **INPUT**

train_data = pd.read_csv("C:/Users/...INSERT FILE PATH HERE.../train.csv")

# LETS TAKE A LOOK AT OUR DATA

train_data.shape
# looks like we have 381k rows and 12 columns

train_data.info()
# from a quick glance it seems we have no null values in our data, 12 different columns with numerical and categorical data.

print(train_data.columns)

train_data.head()

# **DATA CLEANING/EXPLORATORY DATA ANALYSIS**

# delete duplicate columns
# get number of unique values for each column
count = train_data.nunique()

# record columns to delete
to_del = [i for i, v in enumerate(count) if v == 1]

# drop useless columns
train_data.drop(to_del, axis = 1, inplace = True)

train_data.shape
# no duplicate columns to be deleted so no change to the dataframe

# delete duplicate rows
train_data.drop_duplicates(inplace = True)

train_data.shape
# no duplicate rows to be deleted so no change to the dataframe

# check for missing values both numeric features and categorical features
missing = train_data.isnull()
print(missing)

missing_sum = missing.sum()
print(missing_sum)
# no missing values to be dealt with

# VIZUALISATION

# pie chart of males vs female drivers in our data
train_data["Gender"].value_counts().plot(kind = 'pie', autopct = '%.2f')

# line chart of the age distrubtion of our data
sns.kdeplot(train_data["Age"])

# the data above is highly skewed to the right so lets use a FunctionTransformer to try distrbute the data better
# Why - Straight to the point, we use a transformer to reduce the time complexity of a model and increase its accuracy.
sns.kdeplot(FunctionTransformer(func = np.log1p).fit_transform(train_data["Age"]))

# pie chart displaying the number of customers with a drivers license vs without
train_data["Driving_License"].value_counts().plot(kind = "pie", autopct = "%.2f")

# pie chart of previously insured feature
train_data["Previously_Insured"].value_counts().plot(kind = "pie", autopct = "%.2f")
# this looks good, no need to alter

# pie chart of each category for vehicle age
train_data["Vehicle_Age"].value_counts().plot(kind = "pie", autopct = "%.2f")
# no need to alter

# pie chart for vehicle damage
train_data["Vehicle_Damage"].value_counts().plot(kind = "pie", autopct = "%.2f")
# no need to alter

# histogram and boxplot of the annual premium data
plt.rcParams['figure.figsize'] = (10, 5)

plt.subplot(1, 2, 1)
x = train_data["Annual_Premium"]
plt.hist(x, color = "green", edgecolor = "black")
plt.title("Annual Premium histogram")
plt.xticks(rotation = 45)

plt.subplot(1, 2, 2)
sns.boxplot(x, color = "orange")
plt.title("Annual Premium Box Plot")
plt.xticks(rotation = 45)
# plenty of ouliers within the annual premiums with a highly skewed histogram

# **FEATURE EXTRACTION/FEATURE ENGINEERING** 

# change our Gender feature from Male and Female to 1 and 0 respectively
train_data["Gender"] = np.where(train_data["Gender"] == 'Male', 1, 0)

# we replace the original age data with the function transformed data
train_data = pd.concat([FunctionTransformer(func = np.log1p).fit_transform(train_data["Age"]),
                       train_data.drop("Age", axis = 1)] ,
                       axis = 1, join = "inner")

# we want to change vehicle inputs to 1 and 0 rather than a yes and no
train_data["Vehicle_Damage"] = np.where(train_data["Vehicle_Damage"] == 'Yes', 1, 0)

# we need to take care of the outliers in annual premium feature
def delete_outlier(dataset):
    q75, q25 = np.percentile(dataset.dropna(), [75, 25])
    iqr = q75 - q25
    min = q25 - (iqr * 1.5)
    max = q75 + (iqr * 1.5)
    dataset.mask(dataset < min, min, inplace = True)
    dataset.mask(dataset > max, max, inplace = True)

delete_outlier(train_data["Annual_Premium"])

# check corrections
sns.boxplot(train_data["Annual_Premium"], linewidth = 1)
plt.xticks(rotation = 45)
plt.show()


# next I want to seperate the vehicle age feature into 3 features of > 2 years, 1-2 years and < 1 year
# then depending on how old the car is there will be a 1 in the respective column
train_data = pd.concat([pd.get_dummies(train_data["Vehicle_Age"]),
                        train_data.drop("Vehicle_Age", axis = 1)],
                        axis = 1, join = "inner")

# **FEATURE SELECTION/DIMENSIONALITY REDUCTION**
# in order to not confuse the machine learning and feed it too much information we should pick only the most important features

# drop the id column as this won't be neccessary as we have the index already
train_data.drop(['id'], axis = 1, inplace = True)

# as the driver license feature is majority customers with driver license (99.79%) we shall drop this feature
train_data.drop("Driving_License", axis = 1, inplace = True)

train_data.head()

# **Choice of modelling techniques**

# split the data

X = train_data[['1-2 Year', '< 1 Year', '> 2 Years', 'Age', 'Gender', 'Region_Code',
       'Previously_Insured', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']]

Y = train_data[['Response']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

# linear regression model
# linear regression model is used as a bench mark
LR = LinearRegression()
LR.fit(X_train, y_train)

# k nearest neighbour classifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)

# decision tree classifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)

# random forest classifier
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)

# **Hyperparameter Optimisation**
# now we will be performing the tuning of hyperparameters of Random forest model
# the hyperparameters that we will tune includes max_features (number of features used in our dataset) and the n_estimators (number of trees used)

max_features_range = np.arange(1, 12, 1) # there are 11 features in our data
n_estimators_range = np.arange(10, 110, 10) # set number of estimators to begin at 10 - 200 increase in steps of 10
param_grid = dict(max_features = max_features_range, n_estimators = n_estimators_range) # run the instance of the RFC again after this line of code

rand = RandomizedSearchCV(RFC, param_grid, cv = 5)
rand.fit(X_train, y_train)

print("The best parameters are %s with score of %0.2f" % (rand.best_params_, rand.best_score_))

# Initial running of the GridSearchCV ran for over 4 hours
# random search completed with num of trees up to 100 and iterations at 3 in 8mins + (score of 87%)
# second iteration of up to 200 trees and cv = 3 in 22mins + (score of 87%)
# 3rd iteration of up to 100 trees adn cv = 5 in 27 mins + (score of 87%) 


# **Model Evaluation/Comparison**

print("Training set score: {:.3f}".format(LR.score(X_train, y_train)))
print("Training set score: {:.3f}".format(LR.score(X_test, y_test)))

print(KNN, classification_report(y_test, KNN.predict(X_test)))
print(DTC, classification_report(y_test, DTC.predict(X_test)))
print(RFC, classification_report(y_test, RFC.predict(X_test)))
