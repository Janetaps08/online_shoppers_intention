import streamlit as st

st.title('online shoppers intenstion')

st.write('will it create a big revenue')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv['https://github.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/blob/master/online_shoppers_intention.csv']
df

df.columns

df.head()

df.tail()

df.dtypes

df.isna().sum()

df.shape

df.corr(numeric_only=True)

sns.heatmap(df.corr(numeric_only=True))

df['Month'].unique()

a=df['Month'].value_counts()

sns.countplot(x=df['Month'],data=df)
plt.xticks(rotation=45, ha="right")

plt.pie(a.values,labels=a.index,autopct='%1.01f%%')

df['VisitorType'].unique()

plt.xticks(rotation=45, ha="right")
sns.countplot(x=df['VisitorType'],data=df)

df['VisitorType'].value_counts()

df['Revenue'].unique()

sns.countplot(x=df['Revenue'], data=df)

df['Revenue'].value_counts()

df['Administrative'].unique()

df['Administrative'].value_counts()

sns.countplot(x=df['Administrative'],data=df)

df[ 'Administrative_Duration'].unique()

df['Administrative_Duration'].unique()

df['Informational'].unique()

df['Informational'].value_counts()

sns.countplot(x=df['Informational'],data=df)

df['Informational_Duration'].unique()

df['Informational_Duration'].value_counts()

df['ProductRelated'].unique()

df['ProductRelated'].value_counts()

sns.countplot(x=df['ProductRelated'],data=df)

df['ProductRelated_Duration'].unique()

df['ProductRelated_Duration'].value_counts()

df['BounceRates'].unique()

df['BounceRates'].value_counts()

df['ExitRates'].unique()

df['ExitRates'].value_counts()

df['PageValues'].unique()

df['PageValues'].value_counts()

df['SpecialDay'].unique()

df['SpecialDay'].value_counts()

sns.countplot(x=df['SpecialDay'],data=df)

df['OperatingSystems'].unique()

df['OperatingSystems'].value_counts()

sns.countplot(x=df['OperatingSystems'],data=df)

df['Browser'].unique()

df['Browser'].value_counts()

sns.countplot(x=df['Browser'],data=df)

df['Region'].unique()

df['Region'].value_counts()

sns.countplot(x=df['Region'],data=df)

df['TrafficType'].unique()

df['TrafficType'].value_counts()

sns.countplot(x=df['TrafficType'],data=df)

df['Weekend'].unique()

df['Weekend'].value_counts()

sns.countplot(x=df['Weekend'],data=df)

sns.countplot(data=df, x="Weekend", hue="Revenue")
plt.title("Revenue by Weekend")

sns.countplot(data=df, x="ProductRelated", hue="Revenue")
plt.title("Revenue by Product Related")

sns.countplot(data=df, x="SpecialDay", hue="Revenue")
plt.title("Revenue by Special Day")

sns.countplot(data=df, x="Month", hue="Revenue")
plt.title("Revenue by Month")

sns.countplot(data=df, x="OperatingSystems", hue="Revenue")
plt.title("Revenue by Operating Systems")

sns.countplot(data=df, x="Browser", hue="Revenue")
plt.title("Revenue by Browser")

sns.countplot(data=df, x="Region", hue="Revenue")
plt.title("Revenue by Region")

sns.countplot(data=df, x="TrafficType", hue="Revenue")
plt.title("Revenue by Traffic type")

sns.countplot(data=df, x="VisitorType", hue="Revenue")
plt.title("Revenue by Vistor Type")

sns.pairplot(df)

sns.pairplot(df,hue="Revenue")

sns.jointplot(x='Weekend',y='Revenue',data=df,kind='reg')

sns.jointplot(x='OperatingSystems',y='Revenue',data=df,kind='reg')

sns.jointplot(x='SpecialDay',y='Revenue',data=df,kind='reg')

sns.jointplot(x='Month',y='Revenue',data=df,kind='reg')

sns.jointplot(x='TrafficType',y='Revenue',data=df,kind='reg')

sns.boxplot(df['Month'])

sns.boxplot(df['OperatingSystems'])

sns.boxplot(df['Region'])

sns.boxplot(df['TrafficType'])

import matplotlib.pyplot as plt
def removal_box_plot(df,column,threshold):
  sns.boxplot(df[column])
  plt.title(f'original Box plot of{column}')
  plt.show()

  removed_outliers=df[df[column]<= threshold]

  sns.boxplot(removed_outliers[column])
  plt.title(f'Box plot without outliers of {column}')
  plt.show()
  return removed_outliers

threshold_value=4
no_outliers=removal_box_plot(df,'TrafficType',threshold_value)

import matplotlib.pyplot as plt
def removal_box_plot(df,column,threshold):
  sns.boxplot(df[column])
  plt.title(f'original Box plot of{column}')
  plt.show()

  removed_outliers=df[df[column]<= threshold]

  sns.boxplot(removed_outliers[column])
  plt.title(f'Box plot without outliers of {column}')
  plt.show()
  return removed_outliers

threshold_value=6
no_outliers=removal_box_plot(df,'OperatingSystems',threshold_value)


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Month']=lab.fit_transform(df['Month'])
df['VisitorType']=lab.fit_transform(df['VisitorType'])
df['Weekend  ']=lab.fit_transform(df['Weekend'])
df['Revenue']=lab.fit_transform(df['Revenue'])


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x,y)

print('Resampled dataset shape %s' % Counter(y_resampled))

from sklearn.feature_selection import chi2
chi_sqr=chi2(x,y)

chi_sqr = chi2(x, y)
chi_scores = chi_sqr[0]

#higher the chi value higher the importance
features = df.columns[:-1]
plt.xticks(rotation=45, ha="right")
plt.bar(features, chi_scores)

# lower the p value higher the importance
p_value = chi_sqr[1]
features = df.columns[:-1]
plt.xticks(rotation=45, ha="right")
plt.bar(features,p_value)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
s=StandardScaler()
s.fit(x_train)
x_train=s.transform(x_train)
x_test=s.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
knn=KNeighborsClassifier(n_neighbors=7)
naive=BernoulliNB()
sup=SVC()
dt=DecisionTreeClassifier(criterion="entropy")
Rt=RandomForestClassifier(n_estimators=10,criterion="entropy")
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
lst=[knn,naive,sup,dt,Rt]
accuracy_scores = []

for i in lst:
  print("model name:",i)
  i.fit(x_train,y_train)
  y_pred=i.predict(x_test)
  print(y_pred)
  print("***********************************************************************************")
  cm=confusion_matrix(y_test,y_pred)
  print(cm)
  score=accuracy_score(y_test,y_pred)
  print(f'Accuracy Score in {i} is {score}')
  print("***********************************************************************************")
  report=classification_report(y_test,y_pred)
  print(report)
  accuracy_scores.append(score)

import seaborn as sns
sns.barplot(x=lst,y=accuracy_scores)

# from sklearn.model_selection import GridSearchCV
# model1=KNeighborsClassifier()
# parameter={'n_neighbors':[3,5,7,9],'weights':['uniform','distance']}
# gv=GridSearchCV(model1,parameter,cv=10,scoring='accuracy')
# gv.fit(x_train,y_train)

# print(gv.best_params_)

# model2=KNeighborsClassifier(n_neighbors=9,weights='distance')
# model2.fit(x_train,y_train)
# y_pred1=model2.predict(x_test)
# y_pred1

# score1=accuracy_score(y_test,y_pred1)
# score1
