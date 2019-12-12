# -*- coding: utf-8 -*-

"""
1. Loading Libraries / Packages
"""

import os
import json
import pandas as pd
from pandas.io.json import json_normalize 

"""
2. Setting Directory
"""
os.chdir("D:\\Python\\Interview_Data")

"""
3. Loading Json data
"""
j_data=pd.read_json("customersdata.json",lines=True)

"""
4. Extracting data from Json data
"""

# extracting customer data
customer_data=json_normalize(data=j_data['customer'])
customer_data.to_csv("customer_data.csv")

# extracting orders data
orders_data = pd.DataFrame([y for x in j_data['orders'].values.tolist() for y in x])
orders_data.to_csv("orders_data.csv")

# extracting paymentMethods data
paymentMethods_data = pd.DataFrame([y for x in j_data['paymentMethods'].values.tolist() for y in x])
paymentMethods_data.to_csv("paymentMethods_data.csv")

# extracting transations data
transactions_data = pd.DataFrame([y for x in j_data['transactions'].values.tolist() for y in x])
transactions_data.to_csv("transactions_data.csv")

# extracting flag
customer_data_flag= j_data[['fraudulent']].join(customer_data)
customer_data_flag.to_csv("customer_data_flag.csv")

"""
5. Combining data to single data frame for EDA
"""

# merging Transactions and Orders data
final_data=pd.merge(transactions_data,orders_data,'inner',on='orderId')

# merging final data and PaymentMethods data
final_data=pd.merge(final_data,paymentMethods_data,'inner',on='paymentMethodId')
final_data.to_csv("final_data.csv")




