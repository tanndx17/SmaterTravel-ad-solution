
# coding: utf-8

#                                  3. Machine learning: Ad click prediction models

# In[165]:

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, roc_curve
get_ipython().magic(u'matplotlib inline')


# <b>3.1 Feature engineering   ---- what's the main factors for ad clicks?   

# In[31]:

####    Get a n= 10000 sample(limited by computer hardware) and replace null value
data = pd.read_csv('my_data.csv')
s_data = df.sample(n=10000)
s_data.fillna(value = np.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df = importance = pd.DataFrame(imp.fit_transform(s_data),columns=list(data))
X = df.iloc[:,1:]
y = df.iloc[:,0]


# Evaluate feature importance by different models 

# In[92]:

clf = RandomForestClassifier(n_estimators=250,random_state=0)
clf.fit(X, y)

importance = clf.feature_importances_
#importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12,9))
plt.title('Feature Importance:    RandomForestClassifier')
plt.bar(range(X.shape[1]),importance[indices],color = 'r' ,yerr = std[indices], align="center")
plt.xlim([-1,X.shape[1]])
plt.xticks(range(X.shape[1]),indices)
plt.show()


# In[91]:

clf_2 = XGBClassifier()
clf_2.fit(X, y)
#plt.figure(figsize=(15,10))
#plot_importance(clf_2)
#plt.show()

importance = clf_2.feature_importances_
#importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12,9))
plt.title('Feature Importance:  XGBClassifier')
plt.bar(range(X.shape[1]),importance[indices],color = 'b' ,yerr = std[indices], align="center")
plt.xlim([-1,X.shape[1]])
plt.xticks(range(X.shape[1]),indices)
plt.show()


# In[93]:

clf_3 = ExtraTreesClassifier(n_estimators=250,random_state=0)
clf_3.fit(X, y)

importance = clf_3.feature_importances_
#importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12,9))
plt.title('Feature Importance:   ExtraTreesClassifier')
plt.bar(range(X.shape[1]),importance[indices],color = 'g' ,yerr = std[indices], align="center")
plt.xlim([-1,X.shape[1]])
plt.xticks(range(X.shape[1]),indices)
plt.show()


# Top 5 feature across 3 different models
# <br> RandomForest: 15, 24, 25, 32, 16
# <br> Xgboost: 24, 15, 25, 1, 32
# <br> Extra Tree: 15, 32, 24, 1, 6
# <br> The common top feature variables are: 15(historical_clicks), 24(referrer_id), 32(site_language)

# <b>3.2 Ad click prediction

# Model comparison: we will use 4 single models to predict the customers ad clicking
# <br> Support vector classification
# <br>Logistic regression
# <br>Xgboost
# <br>Random Forest

# In[153]:

#skf = StratifiedKFold(n_splits=4)
#for train_index, test_index in skf.split(X,y):   
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

svc_model = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
svc_accuracy = accuracy_score(y_test, y_pred)
svc_precision = precision_score(y_test,y_pred)
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred)
print("Accuracy of SVC: %.2f%%" % (accuracy * 100.0))
print("Precision of SVC: %.2f%%" % (precision * 100.0))
#clf.score(X_test, y_test) 


# In[154]:

xgb_model = xgb.XGBClassifier().fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_precision = precision_score(y_test,y_pred)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred)
print("Accuracy of xgboost: %.2f%%" % (xgb_accuracy * 100.0))
print("Precision of xgboost: %.2f%%" % (xgb_precision * 100.0))


# In[158]:

lg_model = LogisticRegression(C=1, solver='lbfgs').fit(X_train,y_train)
y_pred = lg_model.predict(X_test)
lg_accuracy = accuracy_score(y_test, y_pred)
lg_precision = precision_score(y_test,y_pred)
fpr_lg, tpr_lg, _ = roc_curve(y_test, y_pred)
print("Accuracy of logistic regression: %.2f%%" % (lg_accuracy * 100.0))
print("Precision of logistic regression: %.2f%%" % (lg_precision * 100.0))


# In[156]:

rf_model = RandomForestClassifier(n_estimators = 300 , max_depth =3,random_state = 42).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test,y_pred)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred)
print("Accuracy of random forest: %.2f%%" % (rf_accuracy * 100.0))
print("Precision of random forest: %.2f%%" % (rf_precision * 100.0))


# In[160]:

plt.figure(figsize=(10,8))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lg, tpr_lg, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_svc, tpr_svc, label='Support Vector')
plt.plot(fpr_xgb, tpr_xgb, label='Xgb')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# Looks like xgboost has the best performance across all models. Next we will do 'Parameter optimization' on xgboost

# In[163]:

xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)


# <b>3.3 Make the API model picklable

# In[166]:

pickle.dump(clf, open("best_ad_click.pkl", "wb"))
clf2 = pickle.load(open("best_ad_click.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))


# In[167]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
eval_set=[(X_test, y_test)])


# In[ ]:



