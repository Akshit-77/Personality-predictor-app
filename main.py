#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:


df = pd.read_csv("personality_dataset.csv")
df = df.dropna()


# In[3]:


cat_col = ["Stage_fear", "Drained_after_socializing", "Personality"]
df_cat = df[cat_col]
df_num = df.drop(columns = cat_col)


# In[4]:


df_cat_binary = (pd.get_dummies(df_cat, drop_first = True)).astype(int)
df_cat_binary.head()


# In[5]:


new_df = pd.concat([df_num, df_cat_binary], axis = 1)
new_df.head()


# In[6]:


new_df.describe()


# In[7]:


def split(df, test_size):

    split_index = int(df.shape[0] * (1 - test_size))
    X_train = df.iloc[: split_index, :-1]
    X_test = df.iloc[split_index:, :-1]
    y_train = df.iloc[:split_index, -1]
    y_test = df.iloc[split_index:, -1]
    
    return X_train, X_test, y_train, y_test


# In[8]:


X_train, X_test, y_train, y_test = split(new_df, 0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[9]:


df.info()


# In[10]:


sns.heatmap(new_df.corr(), cmap = 'viridis', annot = True)


# In[11]:


os.makedirs("./models", exist_ok = True)


# In[12]:


from RandomForest import fit_rf_classifier_and_save
fit_rf_classifier_and_save(X_train, y_train, X_test, y_test, n_estimators = 100)


# In[13]:


from NaiveBayes import fit_gnb_classifier_and_save
fit_gnb_classifier_and_save(X_train, y_train, X_test, y_test)


# In[14]:


from LightGBM import fit_lightgbm_and_save
fit_lightgbm_and_save(X_train, y_train, X_test, y_test)


# In[15]:


from LogisticRegression import fit_lr_classifier_and_save
fit_lr_classifier_and_save(X_train, y_train, X_test, y_test)


# In[ ]:




