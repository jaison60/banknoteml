#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import pickle


# In[4]:


data=pd.read_csv("BankNote_Authentication.csv")
print(data)

# In[17]:


x=data.drop('class',axis=1)
y=data['class']


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.30, random_state=123,stratify=y)


# In[25]:


classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)


# In[26]:


y_pred=classifier.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[28]:


print(score)


# In[36]:


pickle_out = open(" BankNote.pickle","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[30]:


classifier.predict([[2,3,4,1]])


# In[ ]:




