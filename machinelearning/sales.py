
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

df = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")

df.head()
df.describe().T

df2.head()


sns.countplot(x=df2['ph'],y=df2['target'])
sns.countplot(x=df2['gravity'],y=df2['target'])
sns.countplot(x=df2['osmo'],y=df2['target'])
sns.countplot(x=df2['cond'],y=df2['target'])
sns.countplot(x=df2['urea'],y=df2['target'])
sns.countplot(x=df2['calc'],y=df2['target'])



df2.groupby('target').mean()
df2.drop(['id'],axis=1)   #X是所有可能的影響變因
X=df2.iloc[:,1:7]

X.head()
Y = df2['target']    #Y是目標值

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=54)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train) 
predictions=lr.predict(X_test)

predictions

from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)



pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predictnot target', 'target'],index=['Truenot target','Truetarget'])

import joblib
joblib.dump(lr,'sales_lr.pkl',compress=3 )
 

