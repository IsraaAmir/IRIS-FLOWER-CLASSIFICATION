#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as seab
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Read Data

# In[4]:


data = pd.read_csv('D:/datascience/Irisflowers/IRIS.csv')


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[ ]:





# Preprocessing The Dataset

# In[8]:


data.isnull().sum()


# In[ ]:





# Exploratory Data Analysis

# In[9]:


data['sepal_length'].hist()


# In[10]:


data['sepal_width'].hist()


# In[9]:


data['petal_length'].hist()


# In[10]:


data['petal_width'].hist()


# Coorelation Matrix

# In[11]:


data.corr()


# In[12]:


corr = data.corr()
fig, ax = plt.subplots(figsize=(5,4))
seab.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[15]:


green_pal = seab.color_palette("viridis", n_colors=3)
seab.pairplot(data,hue='species',palette=green_pal)
plt.show()


# In[17]:


fig,axi = plt.subplots(1,2,figsize=(15,6))
axi[0].plot(data['sepal_length'])
axi[0].plot(data['sepal_width'])
axi[0].set_title('Sepal length vs width')
axi[0].legend(['sepal_length','sepal_width'])
axi[1].plot(data['petal_length'])
axi[1].plot(data['petal_width'])
axi[1].set_title('Petal length vs width')
axi[1].legend(['petal_length','petal_width'])
plt.show()


# In[ ]:





# Converting the cabin column using LabelEncoder

# In[19]:


le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
data


# In[ ]:





# Building mode

# In[27]:


X = data.drop(columns=['species'])
Y = data['species']


# Split Data

# In[28]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[29]:


log = LogisticRegression()

### Fitting my model to the train file

log.fit(x_train,y_train)


# In[30]:


# print metric to get performance
print("Accuracy: ",log.score(x_test, y_test) * 100)


# In[31]:


# decision tree
Des = DecisionTreeClassifier()


# In[32]:


Des.fit(x_train, y_train)


# In[33]:


# print metric to get performance
print("Accuracy: ",Des.score(x_test, y_test) * 100)


# In[ ]:





# In[ ]:




