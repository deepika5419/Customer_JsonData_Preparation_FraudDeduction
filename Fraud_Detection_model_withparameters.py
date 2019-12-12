# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:13:57 2019

@author: Deepika
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import pydot 
import os
import io


os.chdir("D:\\Python\\Interview_Data\\Parameter_DecisionTree")
customer_data=pd.read_csv('customersdata.csv')
#customer_data.index = customer_data.index + 1

# get to know list of features, data shape, stat. description.
customer_data.head()
customer_data.describe()
customer_data.describe(include='all')
print(customer_data.shape)
### Unique columns values
print(customer_data.nunique())
print(customer_data.info())


customer_data['fraudulent'].value_counts()

###plots for visulizing Data

#customer_data['orders__orderId'].value_counts().plot.bar(title='Freq dist of Order Id Type')
customer_data['orders__orderState'].value_counts().plot.bar(title='Freq dist of Order State Type')
customer_data['paymentMethods__paymentMethodType'].value_counts().plot.bar(title='Freq dist of Payment MethodType Type')
customer_data['paymentMethods__paymentMethodProvider'].value_counts().plot.bar(title='Freq dist of Payment Method Provider Type')
customer_data['transactions__transactionFailed'].value_counts().plot.bar(title='Freq dist of Transuction Failed Type')

### Pai Chart  For Getting idea about fraudulent

print("fraudulent as pie chart:")
fig, ax = plt.subplots(1, 1)
ax.pie(customer_data.fraudulent.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'], colors=['yellowgreen','r'])
plt.axis('equal')
plt.ylabel('')

### Correlation between Variabls
corr = customer_data.corr()
# generating correlation heat map
sns.heatmap(corr)

### #Transform categoric to One hot encoding using get_dummies

customer_data_Dummies=pd.get_dummies(customer_data,columns=['orders__orderState','paymentMethods__paymentMethodType','paymentMethods__paymentMethodRegistrationFailure','transactions__transactionFailed'])
customer_data_Dummies.shape
customer_data_Dummies.head(10)
customer_data_Dummies.info()
customer_data_Dummies.describe()

### drop all the non numeric columns

customer_data_Dummies.drop(['transactions__transactionId','transactions__orderId','transactions__paymentMethodId','transactions__transactionAmount','customer__customerEmail','customer__customerPhone','customer__customerDevice','customer__customerBillingAddress','orders__orderId','orders__orderShippingAddress','paymentMethods__paymentMethodId','paymentMethods__paymentMethodProvider','paymentMethods__paymentMethodIssuer','customer__customerIPAddress'],axis=1, inplace=True)

## ### Plot for fraudulent data with respect to orders__orderAmount

plt.hist(customer_data_Dummies['orders__orderAmount'],facecolor='peru',edgecolor='blue',bins=10)
plt.show


### Imputation for missing data
mean_imputer = preprocessing.Imputer()
mean_imputer.fit(customer_data_Dummies[['orders__orderAmount']])
customer_data_Dummies[['orders__orderAmount']] = mean_imputer.transform(customer_data_Dummies[['orders__orderAmount']])

##Replace blanks with NAN in columns fraudulent

customer_data_Dummies['fraudulent'].replace('', np.nan, inplace=True)

### now remove rows have NAN values in fraudulent
customer_data_Dummies.dropna(subset=['fraudulent'], inplace=True)

customer_data_Dummies.to_csv('customer_data_Train.csv',index=True,index_label='ID')

## # Split data into training and test sets

Traindata=pd.read_csv('customer_data_Train.csv')

y = Traindata['fraudulent']
print(y)
X = Traindata[['orders__orderAmount','orders__orderState_failed','orders__orderState_fulfilled','orders__orderState_pending',	'paymentMethods__paymentMethodType_apple pay',	'paymentMethods__paymentMethodType_bitcoin',	'paymentMethods__paymentMethodType_card',	'paymentMethods__paymentMethodType_paypal',	'paymentMethods__paymentMethodRegistrationFailure_False','paymentMethods__paymentMethodRegistrationFailure_True','transactions__transactionFailed_False','transactions__transactionFailed_True']]
print(X)

## Split Data in test and train

X_train,X_test,y_tarin,y_test=train_test_split(X, y,test_size=0.33,random_state=42)


## For validating results
y_test.to_csv('fraudulent_testdata.csv')

### #Build the decision tree model using parameters tuning and grid search cv
tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(2, 8),
}
dt_model=tree.DecisionTreeClassifier()
print(type(tree_param_grid))
dt_grid=model_selection.GridSearchCV(dt_model,tree_param_grid,cv=5,n_jobs=5)
## Fit decision Tree model after paerameter Tuning

model2=dt_grid.fit(X_train,y_tarin.astype(int))

X_test_New = X_test[['orders__orderAmount','orders__orderState_failed','orders__orderState_fulfilled','orders__orderState_pending',	'paymentMethods__paymentMethodType_apple pay',	'paymentMethods__paymentMethodType_bitcoin',	'paymentMethods__paymentMethodType_card',	'paymentMethods__paymentMethodType_paypal',	'paymentMethods__paymentMethodRegistrationFailure_False','paymentMethods__paymentMethodRegistrationFailure_True','transactions__transactionFailed_False','transactions__transactionFailed_True']]


X_test['output']=model2.predict(X_test_New)
tree_performance = accuracy_score(y_test.astype(int), X_test['output'])
print(tree_performance)

###predicted file
X_test.to_csv('Submission_WithParametersTuning.csv')

print('Decision tree:')
print('Best params: ', model2.best_params_)
print('Best scores: ', model2.best_score_)
print('My score: ', tree_performance)





