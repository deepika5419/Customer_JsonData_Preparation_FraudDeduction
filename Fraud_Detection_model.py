# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:40:32 2019

@author: Deepika
"""


import pandas as pd
import os
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
import io
import seaborn as sns


os.chdir("D:\\Python\\Interview_Data\\")


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

### plotings for visulizing Data features

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

### Plot for fraudulent data with respect to orders__orderAmount
sns.relplot(x='fraudulent', y='orders__orderAmount',kind='line', data=customer_data_Dummies)
plt.hist(customer_data_Dummies['orders__orderAmount'],facecolor='peru',edgecolor='blue',bins=10)
plt.show

### Imputation for handing missing values
mean_imputer = preprocessing.Imputer()
mean_imputer.fit(customer_data_Dummies[['orders__orderAmount']])
customer_data_Dummies[['orders__orderAmount']] = mean_imputer.transform(customer_data_Dummies[['orders__orderAmount']])

##Replace blanks with NAN in columns fraudulent

customer_data_Dummies['fraudulent'].replace('', np.nan, inplace=True)

### now remove rows have NAN values in fraudulent
customer_data_Dummies.dropna(subset=['fraudulent'], inplace=True)

##After finalizing variables and data  
customer_data_Dummies.to_csv('D:\\Python\\Interview_Data\\customer_data_Train.csv',index=True,index_label='ID')


## # Split data into training and test sets

Traindata=pd.read_csv('D:\\Python\\Interview_Data\\customer_data_Train.csv')

y = Traindata['fraudulent']
print(y)
X = Traindata[['orders__orderAmount','orders__orderState_failed','orders__orderState_fulfilled','orders__orderState_pending',	'paymentMethods__paymentMethodType_apple pay',	'paymentMethods__paymentMethodType_bitcoin',	'paymentMethods__paymentMethodType_card',	'paymentMethods__paymentMethodType_paypal',	'paymentMethods__paymentMethodRegistrationFailure_False','paymentMethods__paymentMethodRegistrationFailure_True','transactions__transactionFailed_False','transactions__transactionFailed_True']]
print(X)

#### Split Data in to train and test
X_train,X_test,y_tarin,y_test=train_test_split(X, y,test_size=0.33,random_state=42)

## For validating results
y_test.to_csv('D:\\Python\\Interview_Data\\fraudulent_testdata.csv')

### Build Decision Tree Model
dt=tree.DecisionTreeClassifier()

model=dt.fit(X_train,y_tarin.astype(int))

X_test['Output'] = model.predict(X_test)

X_test.to_csv('D:\\Python\\Interview_Data\\Submission_Decision_Tree.csv')

accuracy_score(y_test.astype(int), X_test['Output'])

## Accuracy Matrix

pd.DataFrame(
        confusion_matrix(y_test.astype(int), X_test['Output']),
        columns=['Yes','No'],
        index=['Yes','NO'])
##Decision Tree

objStringIO = io.StringIO() 
tree.export_graphviz(dt, out_file = objStringIO, feature_names = X_train.columns)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#Use out_file = objStringIO to getvalues()
file1 = pydot.graph_from_dot_data(objStringIO.getvalue())[0]
#os.chdir("D:\\Data Science\\Data\\")
file1.write_pdf("DecissionTree.pdf")







## Data cleaning Task

#### Customer Email-Id
#
#
#
#def check_mail(customer_data):
#    validcode = (customer_data['customer__customerEmail'].str.contains('@')) & (customer_data['customer__customerEmail'].str.contains('.org', case= False) & (customer_data['customer__customerEmail'].str.contains('sales', case=False)))
#    return validcode
#        
#def highlight_email(s):
#    if check_mail(customer_data).all():
#        color = ''
#    else:
#        color = 'red'
#    return 'background-color: %s' % color
#
#customer_data.style.applymap(highlight_email, ['customer__customerEmail'])
#
#
#
#customer_data['is_valid_email'] = customer_data['customer__customerEmail']