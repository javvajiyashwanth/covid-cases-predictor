#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY PACKAGES

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[31]:


import pickle
import json


# # READING CSV FILE

# In[32]:


df = pd.read_csv("covid-data.csv")
df.head()


# In[33]:


data = df.groupby(['continent', 'location', 'population'])
data.first()


# # DATA PREPROCESSING

# In[34]:


df.isnull().sum()


# In[35]:


df = df.dropna()


# In[36]:


df = pd.concat([df, pd.get_dummies(df.location)], axis = 'columns')
df.head()


# In[37]:


df = pd.concat([df, pd.get_dummies(df.continent)], axis = 'columns')
df.head()


# In[38]:


df = df.drop(['continent', 'location'], axis = 'columns')
df.head()


# In[39]:


X = df.drop(['total_cases', 'new_cases', 'total_deaths', 'new_deaths'], axis = 'columns')
X.head()


# In[40]:


keys = data.groups.keys()
data = {
    'columns': [col for col in X],
    'data': {}
}
for (continent, country, population) in keys: 
    temp = data['data'].get(continent, {})
    temp[country] = population
    data['data'][continent] = temp
with open('data.json', 'w') as f:
    f.write(json.dumps(data))


# # NEW CASES MODEL

# In[41]:


y = df.new_cases
y.head()


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)


# In[43]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[44]:


with open('new_cases_model.pickle', 'wb') as f:
    pickle.dump(lr, f)


# # TOTAL CASES MODEL

# In[45]:


y = df.total_cases
y.head()


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)


# In[47]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[48]:


with open('total_cases_model.pickle', 'wb') as f:
    pickle.dump(lr, f)


# # NEW DEATHS MODEL

# In[49]:


y = df.new_deaths
y.head()


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)


# In[51]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[52]:


with open('new_deaths_model.pickle', 'wb') as f:
    pickle.dump(lr, f)


# # TOTAL DEATHS MODEL

# In[53]:


y = df.total_deaths
y.head()


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)


# In[55]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[56]:


with open('total_deaths_model.pickle', 'wb') as f:
    pickle.dump(lr, f)

