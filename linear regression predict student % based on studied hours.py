#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[48]:


#predict the percentage of a student based on the no. of study hours


#reading the data
df = pd.read_csv(r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print("successfully imported data into console")


# In[49]:


#view data
df.head(26)


# In[50]:


df.describe()


# In[36]:


df.info()


# In[51]:


df.shape


# In[52]:


#enter distribution scores and plot them. in the bellow scatter we see the colleration between scores and hours studied
df.plot(x="Hours", y="Scores", style="o") #indicates that the data points will be represented as circles. The use of circles as markers typically indicates a scatter plot. if i want to make line plot remove style
plt.title("hours vs percentage")
plt.xlabel("the hours studied")
plt.ylabel("the percentage score")
plt.show() # is a function from matplotlib.pyplot that displays the plot on the screen.


# In[53]:


# Split the dataset into features (X) and target variable (y)
# The .iloc indexer in pandas is used to select data from a DataFrame by integer-based indexing, rather than by label-based indexing. It allows you to access specific rows or columns of a DataFrame using integer-based positions.
x = df.iloc[:, :-1].values # .value: converts the selected DataFrame slice into a NumPy array, extracting the values from the DataFrame.
y = df.iloc[:, :1].values


# In[54]:


#random split. we use scikit learn method with train-test_split helperfunction
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split #This function is commonly used in machine learning to split a dataset into training and testing sets. 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# x, y The feature and target variables, respectively, which will be split into training and testing sets.
# test_size=0.2: Specifies the proportion of the dataset to be allocated for testing. In this case, 20% of the data will be used for testing, and 80% will be used for training.
# random_state=0 in one run, using random_state=0 again in a different run will yield the same split. if i put no. it will result to different splits

#The function returns four arrays: x_train, x_test, y_train, and y_test.
# x_train and y_train represent the feature and target variables, respectively, for the training set.
# x_test and y_test represent the feature and target variables, respectively, for the testing set.


# In[55]:


from sklearn.linear_model import LinearRegression

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(x_train, y_train)

print("trainning completed")


# In[56]:


#implementing the plotting testing data
line = model.coef_*x+model.intercept_
plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# In[57]:


#predicting percentage of marks
y_pred = model.predict(x_test)
prediction = pd.DataFrame({'Hours': [i[0] for i in x_test], 'predicted marks': [k for k in y_pred]})
prediction

#[i[0] for i in x_test] extracts the hours studied values from x_test and forms a list.
#[k for k in y_pred] forms a list containing the predicted marks.


# In[58]:


# Compare the predicted values with the actual values
df = pd.DataFrame({"Actual": y_test.ravel(), "Predicted": y_pred.ravel()}, index=range(len(y_test)))
print(df)

# range(len(y_test)), which generates an index based on the length of y_test. This ensures that each row in the DataFrame has a corresponding index value. because o the error The error message "ValueError: If using all scalar values, you must pass an index" typically occurs when trying to create a DataFrame with scalar values (single values) 
# .ravel() method to convert them to 1-dimensional arrays.


# In[59]:


plt.scatter(x=x_test, y=y_test, color="blue") #This line creates a scatter plot to represent the actual values.
plt.plot(x_test, y_pred, color="black") # This line creates a line plot to represent the predicted values.
plt.title("actual vs predicted", size=20)
plt.xlabel("hours studied", size=12)
plt.ylabel("marks percentage", size=12)
plt.show()


# In[60]:


#what will be the predicted score of student if he/she studies for 9.25 hours/day?
hours= [[9.25]]  #The nested list is used because the .predict() method of the linear regression model expects a 2-dimensional array-like input.
own_pred = model.predict(hours)
print("no.of hours = {}".format(hours))
print("prediction_score = {}".format(own_pred[0]))


# In[61]:


from sklearn import metrics
print("mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




