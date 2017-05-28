
# coding: utf-8

# In[11]:

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import xgboost as xgb
get_ipython().magic(u'matplotlib inline')


# In[13]:

df = pd.read_csv('my_data.csv')
s_data = df.sample(n=10000)
s_data.fillna(value = np.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df = importance = pd.DataFrame(imp.fit_transform(s_data),columns=list(data))


# In[8]:

y = df.iloc[:,1]
X = df.drop('ad_click_revenue',axis=1)


# In[47]:

clf = RandomForestRegressor(n_estimators=300,random_state=0)
clf.fit(X, y)

importance = clf.feature_importances_
#importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12,9))
plt.title('Feature Importance:    RandomForestRegressor')
plt.bar(range(X.shape[1]),importance[indices],color = 'r' ,yerr = std[indices], align="center")
plt.xlim([-1,X.shape[1]])
plt.xticks(range(X.shape[1]),indices)
plt.show()


# In[46]:

#print ("Top factors that affects revenue:"%s %s" % (list(X)[1],list(X)[22]))
print('The top factos the affect revenue are:')
print (', '.join([list(X)[1], list(X)[22],list(X)[13],list(X)[12],list(X)[24]]))


# In[ ]:



