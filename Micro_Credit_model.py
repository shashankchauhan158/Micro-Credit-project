#!/usr/bin/env python
# coding: utf-8

# # Project Name: Micro Credit Model

# # Predicting in terms of a probability for each loan transaction, whether the customer will be paying back the loaned amount within 5 days of insurance of loan using machine learning
# 
# 
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether the customer will be paying back the loaned amount within 5 days of insurance of loan in time or not.
# 
# 1. **Problem Definition**
# 
# In a statement,
# The problem we will be exploring is binary classification (a sample can only be one of two things).
# 
# This is because we're going to be using a number of differnet features (pieces of information) about a person to predict in terms of a probability for each loan transaction, whether the customer will be paying back the loaned amount within 5 days of insurance of loan. In this case, Label ‘1’ indicates that the loan has been payed i.e. Non- defaulter, while, Label ‘0’ indicates that the loan has not been payed i.e. defaulter.
# 
# 2. **Data**
# 
# All of the dataset values were provided by a client.
# 
# 
# 3. **Evaluation**
# 
# Evaluating a models predictions using problem-specific evaluation metrics
# 
# 
# 4. **Features**
# 
# The following are the features we'll use to predict our target variable (Label ‘1’ indicates that the loan has been payed i.e. Non- defaulter, while, Label ‘0’ indicates that the loan has not been payed i.e. defaulter).
# 
# * label : Flag indicating whether the user paid back the credit amount within 5 days of issuing the loan{1:success, 0:failure}
# * msisdn : mobile number of user
# * aon : age on cellular network in days
# * daily_decr30: Daily amount spent from main account, averaged over last 30 days (in Indonesian Rupiah)
# * daily_decr90: Daily amount spent from main account, averaged over last 90 days (in Indonesian Rupiah)
# * rental30: Average main account balance over last 30 days
# * rental90: Average main account balance over last 90 days
# * last_rech_date_ma: Number of days till last recharge of main account
# * last_rech_date_da: Number of days till last recharge of data account
# * last_rech_amt_ma: Amount of last recharge of main account (in Indonesian Rupiah)
# * cnt_ma_rech30: Number of times main account got recharged in last 30 days
# * fr_ma_rech30: Frequency of main account recharged in last 30 days
# * sumamnt_ma_rech30: Total amount of recharge in main account over last 30 days (in Indonesian Rupiah)
# * medianamnt_ma_rech30: Median of amount of recharges done in main account over last 30 days at user level (in Indonesian Rupiah)
# * medianmarechprebal30: Median of main account balance just before recharge in last 30 days at user level (in Indonesian Rupiah)
# * cnt_ma_rech90: Number of times main account got recharged in last 90 days
# * fr_ma_rech90: Frequency of main account recharged in last 90 days
# * sumamnt_ma_rech90 : Total amount of recharge in main account over last 90 days (in Indian Rupee)
# * medianamnt_ma_rech90: Median of amount of recharges done in main account over last 90 days at user level (in Indian Rupee)
# * medianmarechprebal90: Median of main account balance just before recharge in last 90 days at user level (in Indian Rupee)
# * cnt_da_rech30: Number of times data account got recharged in last 30 days
# * fr_da_rech30: Frequency of data account recharged in last 30 days
# * cnt_da_rech90: Number of times data account got recharged in last 90 days
# * fr_da_rech90: Frequency of data account recharged in last 90 days
# * cnt_loans30: Number of loans taken by user in last 30 days
# * amnt_loans30: Total amount of loans taken by user in last 30 days
# * maxamnt_loans30: maximum amount of loan taken by the user in last 30 days
# * medianamnt_loans30: Median of amounts of loan taken by the user in last 30 days
# * cnt_loans90: Number of loans taken by user in last 90 days
# * amnt_loans90: Total amount of loans taken by user in last 90 days
# * maxamnt_loans90: maximum amount of loan taken by the user in last 90 days
# * medianamnt_loans90: Median of amounts of loan taken by the user in last 90 days
# * payback30: Average payback time in days over last 30 days
# * payback90: Average payback time in days over last 90 days
# * pcircle: telecom circle
# * pdate: date
# 

# In[2]:


# Importing libraries for data loading and visualization..
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from nltk import flatten

import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Loading the dataset..
df_credit=pd.read_csv("Data file.csv",parse_dates=['pdate'],index_col=None)
df_credit


# In[5]:


# checking the features, their shape, duplicate values and nan values in the Datasets

print("\nFeatures Present in the Dataset: \n", df_credit.columns)
shape=df_credit.shape
print("\nTotal Number of Rows : ",shape[0])
print("Total Number of Features : ", shape[1])
print("\n\nData Types of Features :\n", df_credit.dtypes)
print("\nDataset contains any NaN/Empty cells : ", df_credit.isnull().values.any())
print("\nTotal number of empty rows in each feature:\n", df_credit.isnull().sum(),"\n\n")
print("Total number of unique values in each feature:")
for col in df_credit.columns.values:
    print("Number of unique values of {} : {}".format(col, df_credit[col].nunique()))
    
print ('\nCreditor and defaulter counts','\n',df_credit.label.value_counts())


# In[6]:


# Checking Statistical Informations...
df_credit.describe()


# **Some features even have negative values like the age on cellular network, main account last recharge date, data account last recharge date. Negative values in these features make no sense thus these values should be removed(Shown in EDA section).** 

# In[7]:


# Dropping those features which are not adding any important information...
df_credit.drop(['Unnamed: 0','pcircle','msisdn'],axis=1,inplace=True)
df_credit


# # EDA (Exploratory data Analysis)
# 

# In[8]:


# Checking for negative values in the Dataset, as we can see that many of the features are having negative values...
(df_credit.drop(['pdate'],axis=1) >= 0).all()


# **Some of the features like "rental30" and "rental90" can have negative values as these feature will show the loan amount per user.**
# 
# 

# In[9]:


# Dropping few features...
df_credit.drop(['rental30','rental90','pdate'],axis=1,inplace=True)


# In[10]:


# This loop will drop all the negative values from those features in which they are not needed...
index=[]
for cols in df_credit.columns.values:
    Index_1=df_credit[df_credit[f'{cols}'] < 0].index.values
    Index_2=Index_1.tolist()
    index.append(Index_2)
index_fl=flatten(index)
set(index_fl)
len(index_fl)


# In[11]:


# Dropping the negative values in the features where these negative values don't make any sense...
df_credit.drop(index_fl,inplace=True)


# In[12]:


# Checking if the negative value dropping process is succesful or not..
(df_credit >= 0).all(0)


# In[13]:


# Checking some rows where negative values were present, as we can see there are no negative values..
df_credit[20:30]


# In[14]:


# reading dataset...
df_credit_new=pd.read_csv("Data file.csv",parse_dates=['pdate'],index_col=None)


# In[15]:


# Now placing the dropped "rental30" and "rental90" values in which the negative values are not outliers...
df_credit['rental30']=df_credit_new['rental30']
df_credit['rental90']=df_credit_new['rental90']
df_credit['pdate']=df_credit_new['pdate']


# In[16]:


# Displaying few rows for checking whether the insertion is succesful or not..
df_credit[20:30]


# In[17]:


# Checking the correlation between the features and the label...
df_credit.corr()


# # Univariant Plot Analysis
# 

# In[18]:


# For loop to display some important features counts in one go...
list=['label', 'last_rech_amt_ma', 'cnt_ma_rech30','cnt_ma_rech90', 'fr_ma_rech90',
       'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30','amnt_loans30',
      'medianamnt_loans30', 'amnt_loans90', 'maxamnt_loans90', 'medianamnt_loans90', ]
for i in list:
    plt.subplots(figsize=(20,8))
    sns.countplot(i,data=df_credit)
    plt.show()


# In[19]:


# Plotting the boxplot in order to check few statistical values and outliers
df_credit.drop('pdate',axis=1).plot(kind='box', subplots=True, layout=(12,3),figsize=(20,20), grid=True, notch=True, color='red',legend=True)


# As we can see that except the negative values there are still lot of outliers present.
# 
# 

# In[20]:


# Checking the Distribution using the histogram plot.
df_credit.hist(figsize=(20,20),grid=True,layout=(5,7),bins=30,color='lightblue')


# **From the above histogram plots, it is clear that the data is rightly skewed...
# 
# 

# In[21]:


# Dropping the date feature to proceed further for outliers removing part...
df_credit.drop('pdate',axis=1,inplace=True)


# In[22]:


# Checking correlation among the features...
df_credit.corr()


# # Note:-
# 
# **The Dataset we are having, consists of some features giving information anout the user for the time span of 30 days and 90 days. According to me if we have data of large number of days for a particular user then we could interpret User's behavior more precisely because many users have the tendancy of repeating the same things. Thus the features having the data with a time span of 90 days gives more information about the user as compared to the features with a time span of 30 days.**
# 
# **From the above correlation table it is also clear that the features with time span of 30 and 90 days almost hav ethe same correlation thus we can drop one for the same information.**

# In[24]:


# Now dropping the features having same correlation...
df_credit.drop(["daily_decr30","fr_ma_rech30","payback30","rental30","medianamnt_loans30","amnt_loans30",
                "fr_da_rech30","cnt_da_rech30","sumamnt_ma_rech30","fr_ma_rech30","cnt_ma_rech30"],axis=1,inplace=True)


# In[25]:


# Checking the dataframe after dropping...
df_credit


# **Using MS EXCEL I have found the maximum values a feature can have, beyond these values the values are unimaginable.**

# **(for an example beyond the value [2500], the very next value in "aon" feature comes out to be around 2379 years, which means a user is using the telephone services from 359 BCE which is clearly not possible).**

# In[27]:


#************** Threshold Values that some of the important feature can have according to the data provided ***********

# 1) rental30 and rental90 can be negative 
# 2) last_rech_date_ma                    ==> max 113
# 3) last_rech_date_da                    ==> max 115
# 4) aon                                  ==> less than 2500  
# 5) fr_ma_rech30                         ==> less than 38
# 6) maxamnt_loans30                      ==> less than 12
# 7) cnt_loans90                          ==> less than 71


# In[28]:


# Removing outliers from features...
df_clean = df_credit[df_credit['last_rech_date_ma'] < 250]  
df_clean = df_clean[df_clean['last_rech_date_da'] <= 115]
df_clean = df_clean[df_clean['aon'] < 2500]
df_clean = df_clean[df_clean['cnt_loans90'] <= 71]
df_clean = df_clean[df_clean['maxamnt_loans30'] <= 12]


# In[29]:


# Checking the clean dataset...
df_clean


# In[31]:


df_clean.skew()


# # A lot of skewness is present in the data, thus removing it.
# 
# 

# In[32]:


# Removing Skewness.........
for i in df_clean.drop(['label','rental90'],axis=1).columns:
    if df_clean.skew().loc[i]>0.55:
        df_clean[i]=np.log1p(df_clean[i])


# In[33]:


# Checking skewness again...
df_clean.skew()


# **Visualization after outliers removal**
# 

# In[34]:


# Checking the unique values in the features...
df_clean.nunique()


# In[37]:


# For loop to display some important features counts in one go after outlier removal...
list=['label', 'last_rech_amt_ma','cnt_ma_rech90', 'fr_ma_rech90','cnt_da_rech90', 'fr_da_rech90',
      'cnt_loans30', 'amnt_loans90','maxamnt_loans30', 'medianamnt_loans90', ]


for i in list:
    plt.subplots(figsize=(20,8))
    sns.countplot(i,data=df_credit)
    plt.xticks(rotation=90,fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel(i,fontsize=20)
    plt.xlabel(f'{i}',fontsize=20)
    plt.show()


# In[38]:


# Checking the Distribution using the histogram plot.
df_clean.hist(figsize=(20,20),grid=True,layout=(5,7),bins=30,color='lightblue') 


# **From the above plots it is clear that the data is now normally distributed after outlier treatment and skewness removal.**
# 
# 

# In[40]:


# Checking feature information...
df_clean.info()


# ((209593-197074)/209593)*100= 5.97%
# 
# **After all this cleaning process it is clear that only 5.97% data is removed which was not imaginable for the information these features are providing.**

# In[41]:


# checking statistical feature information...
df_clean.describe()


# In[42]:


# printing cleaned data...
df_clean


# # Bivariant Analysis
# 

# **All the categories that is being made to make the visualizations easy are solemnly based on the Description i.e statistical summary of the data plotted above. for instance low comes under(0-25%), average comes under(25-75%) and high comes over 75% of the data values in a given feature**

# In[44]:


# Making a copy of cleaned data for the visualization purpose...
df_visual=df_clean.copy()


# In[45]:


# Dropping the features which will not be used for visualization purplose...
df_visual.reset_index(inplace = True)
df_visual.drop(['daily_decr90', 'last_rech_date_ma','last_rech_date_da', 'last_rech_amt_ma', 'medianamnt_ma_rech30',
                'medianmarechprebal30', 'fr_ma_rech90','medianamnt_ma_rech90', 'medianmarechprebal90',
                'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30', 'maxamnt_loans30','maxamnt_loans90', 'medianamnt_loans90',]
               ,axis=1,inplace=True)


# In[46]:


# printing the features used for visualization...
df_visual


# **Feature "rental90": Average main account balance over last 90 days vs Loan Repayment Percentage within 5 days**

# In[47]:


# Making a new  feature "Balance_Category" to store the different categories for the rental90 feature to get a better view of the visualtion..
conditions_1=[(df_visual['rental90'] <=0),df_visual['rental90'].between(0,1379),df_visual['rental90'].between(1379,4280),(df_visual['rental90'] > 4280)]
values_1= ['Negative or zero Balance', 'Low Balance', 'Average Balance','High Balance']
df_visual['Balance_Category']=np.select(conditions_1,values_1)


# In[48]:


# Printing the new feature...
df_visual['Balance_Category'].value_counts()


# **According to the data it is clear that, users having Low balance are more in number and the persons with negative or zero balance are less.**
# 
# 

# In[49]:


# Mapping "Balance_Category" feature with precentage value with respect to the label. 
balance_category_percent = pd.crosstab(df_visual['label'],df_visual['Balance_Category']).apply(lambda x: x/x.sum()*100)
balance_category_percent = balance_category_percent.transpose()


# In[50]:


# printing values...
balance_category_percent


# **Label 0: defaulter**
#     
# **Label 1: Non-defaulter**
# 
#    
# 
# 

# In[52]:


#   Graphical representation of the User's balance along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.

balance_category_percent.plot(kind='bar',color='rgbymck',figsize=(15,5))
plt.title('Average main account balance over last 90 days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Balance Category',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above Graph and the crosstab table it is clear that:
# 
# 1) 28% of Users having negative or zero balance are defaulters, which is very high.
# 2) 10% to 12% Users are defaulters which falls in the category of Average and Low balance category.
# 3) Users having high balance and are defaulters are very less in number.

# **Feature "cnt_loans90": Number of loans taken by user in last 90 days vs Loan Repayment Percentage within 5 days**
# 

# In[53]:


# Making a new  feature "Loans_Frequency" to store the different categories for the cnt_loans90 feature to get a better view of the visualtion..
conditions_2=[(df_visual['cnt_loans90'] <=0),df_visual['cnt_loans90'].between(0,2),(df_visual['cnt_loans90'] > 2)]
values_2= ['No Loans Taken', 'Average number of loans Taken','Too much loans taken']
df_visual['Loans_Frequency']=np.select(conditions_2,values_2)


# In[54]:


df_visual['Loans_Frequency'].value_counts()


# 
# **Users who take average amount of loans more in number.**
# 
# 

# In[56]:


# Mapping Loans_Frequency with precentage value with respect to label 
Loans_Frequency_percent = pd.crosstab(df_visual['label'],df_visual['Loans_Frequency']).apply(lambda x: x/x.sum()*100)
Loans_Frequency_percent = Loans_Frequency_percent.transpose()
Loans_Frequency_percent


# In[57]:


#   Graphical representation of the Loans_Frequency along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.

Loans_Frequency_percent.plot(kind='bar',figsize=(15,5))
plt.title('Number of loans taken by user in last 90 days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Loans Frequency',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above graph it is clear that:
# 
# 1) Users who take more number of loans are non defaulters(i.e 98% of the category) as they repays the loan within the given time i.e 5 days.
# 2) 14% of the Users are are among the average number of loan taken category are defaulters.

# **Feature "sumamnt_ma_rech90":Total amount of recharge in main account over last 90 days (in Indian Rupee) vs Loan Repayment Percentage within 5 days**
# 

# In[61]:


# Making a new  feature "Recharge_Amount_Category" to store the different categories for the sumamnt_ma_rech90 feature to get a better view of the visualtion..
conditions_3=[(df_visual['sumamnt_ma_rech90'] <=0),df_visual['sumamnt_ma_rech90'].between(0,12),df_visual['sumamnt_ma_rech90'].between(12,15),(df_visual['sumamnt_ma_rech90'] > 14)]
values_3= ['No Recharge', 'Between 0 and 12(Rupiah)', 'Between 12 and 15(Rupiah)','More than 15']
df_visual['Recharge_Amount_Category']=np.select(conditions_3,values_3)


# In[62]:


df_visual['Recharge_Amount_Category'].value_counts()


# In[63]:


# Mapping Recharge_Amount_Category with precentage value with respect to label 
Recharge_Amount_Category_percent = pd.crosstab(df_visual['label'],df_visual['Recharge_Amount_Category']).apply(lambda x: x/x.sum()*100)
Recharge_Amount_Category_percent = Recharge_Amount_Category_percent.transpose()
Recharge_Amount_Category_percent


# In[64]:


#   Graphical representation of the Recharge_Amount_Category along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.


Recharge_Amount_Category_percent.plot(kind='bar',color='rgbymck',figsize=(15,5))
plt.title('Total amount of recharge in main account over last 90 days (in Indian Rupee) vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Recharge_Amount_Category',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above graph it is clear that:
# 
# 1) 40 % of the Users who do not even recharged in the 90 days are defaulters only.
# 2) Users who do very high amount of recharge always pays their loans on time. i.e 98% of them are non defaulters.
# 3) 34% of the Users who do less amount of recharge are defaulters.

# **Feature "payback90":Average payback time in days over last 90 days vs Loan Repayment Percentage within 5 days**
# 

# In[65]:


# Making a new  feature "Defaulters_Category" to store the different categories for the payback90 feature to get a better view of the visualtion..
conditions_4=[(df_visual['payback90'] <=5),(df_visual['payback90'] > 5)]
values_4= ['Not Defaulters','Defaulters']
df_visual['Defaulters_Category']=np.select(conditions_4,values_4)


# In[66]:


df_visual['Defaulters_Category'].value_counts()


# In[67]:


# Mapping Defaulters_Category with precentage value with respect to label 
Defaulters_Category_percent = pd.crosstab(df_visual['label'],df_visual['Defaulters_Category']).apply(lambda x: x/x.sum()*100)
Defaulters_Category_percent = Defaulters_Category_percent.transpose()
Defaulters_Category_percent


# In[68]:


#   Graphical representation of the Defaulters_Category along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.


Defaulters_Category_percent.plot(kind='bar',figsize=(15,5))
plt.title('Average payback time in days over last 90 days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Defaulters_Category',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# **Feature "amont_loans90": Total amount of loans taken by user in last 90 days vs Loan Repayment Percentage within 5 days**
# 

# In[69]:


# Making a new  feature "Loan_Amount_Category" to store the different categories for the amnt_loans90 feature to get a better view of the visualtion..
conditions_5=[(df_visual['amnt_loans90'] <=0),df_visual['amnt_loans90'].between(0,1),df_visual['amnt_loans90'].between(1,3),(df_visual['amnt_loans90'] > 3)]
values_5= ['No loans', 'Low Amount', 'Average Amount','High Amount']
df_visual['Loan_Amount_Category']=np.select(conditions_5,values_5)


# In[70]:


# Printing the values...
df_visual['Loan_Amount_Category'].value_counts()


# **Users who take small loans are more in number**
# 
# 

# In[71]:


# Mapping Loan_Amount_Category with precentage value with respect to label 
Loan_Amount_Category_percent = pd.crosstab(df_visual['label'],df_visual['Loan_Amount_Category']).apply(lambda x: x/x.sum()*100)
Loan_Amount_Category_percent = Loan_Amount_Category_percent.transpose()
Loan_Amount_Category_percent


# In[72]:


#   Graphical representation of the Loan_Amount_Category along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.

Loan_Amount_Category_percent.plot(kind='bar',color='rgbymck',figsize=(15,5))
plt.title('Total amount of loans taken by user in last 90 days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Loan Amount Category',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above graph it is clear that:
# 
# 1) Users who did not take any loans are non defaulters.
# 2) Most of the Users(i.e 97%) who take large amount of loans comes under non defaulter category.
# 3) 17% of the users who take small loans are defaulters.
# 
# 

# **Feature "cnt_ma_rech90" :Number of times main account got recharged in last 90 days vs Loan Repayment Percentage within 5 days**
# 

# In[73]:


# Making a new  feature "Recharge Frequency" to store the different categories for the cnt_ma_rech90 feature to get a better view of the visualtion..
conditions_6=[(df_visual['cnt_ma_rech90'] <=0),df_visual['cnt_ma_rech90'].between(0,1),df_visual['cnt_ma_rech90'].between(1,3),(df_visual['cnt_ma_rech90'] > 3)]
values_6= ['Not Recharged', 'Low Recharge Frequency', 'Average Recharge Frequency','High Recharge Frequency']
df_visual['Recharge Frequency']=np.select(conditions_6,values_6)


# In[74]:


# Printing values...
df_visual['Recharge Frequency'].value_counts()


# In[75]:


# Mapping Recharge Frequency with precentage value with respect to label 
Recharge_Frequency_percent = pd.crosstab(df_visual['label'],df_visual['Recharge Frequency']).apply(lambda x: x/x.sum()*100)
Recharge_Frequency_percent = Recharge_Frequency_percent.transpose()
Recharge_Frequency_percent


# In[76]:


#   Graphical representation of the Recharge Frequency along with their (defaulter or non defaulter) category and 
#                    their ability to repay the loan amount within 5 days.

Recharge_Frequency_percent.plot(kind='bar',figsize=(15,5))
plt.title('Number of times main account got recharged in last 90 days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Recharge Frequency',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above graph it is clear that:
# 
# 1) Among the Users who have not done a single recharge in 3 months 40% are defaulters.
# 2) Among the Users who are very frequent in recharging and who always pay their loans on time are more in number i.e 99% of the total category, which is a good news for the company.

# **Feature "aon": age on cellular network in days vs Loan Repayment Percentage within 5 days**

# In[77]:


# Making a new  feature "Users_Category" to store the different categories for the "aon" feature to get a better view of the visualtion..
conditions_7=[(df_visual['aon'] <2),df_visual['aon'].between(2,5),(df_visual['aon'] > 5)]
values_7= ['New Users','Average Users','Old Users']
df_visual['Users_Category']=np.select(conditions_7,values_7)


# In[78]:


# Printing the values...
df_visual['Users_Category'].value_counts()


# **New Users are very few in number as compared to Old Users which are in a large number**
# 
# 

# In[80]:


# Mapping Users_Category with precentage value with respect to label.. 
Users_Category_percent = pd.crosstab(df_visual['label'],df_visual['Users_Category']).apply(lambda x: x/x.sum()*100)
Users_Category_percent = Users_Category_percent.transpose()
Users_Category_percent


# In[81]:


#         Graphical representation of the User's Age on the cellular network along with 
#    their (defaulter or non defaulter) category and their ability to repay the loan amount within 5 days...

Users_Category_percent.plot(kind='bar',color='rgbymck',figsize=(15,5))
plt.title('age on cellular network in days vs Loan Repayment Percentage within 5 days',fontsize=15)
plt.ylabel('Loan Repayment Percentage within 5 days',fontsize=15)
plt.xlabel('Users Category',fontsize=15)
plt.xticks(rotation = 'horizontal',fontsize=12)


# # Conclusion:
# From the above graph it is clear that:
# 
# 1) 32% of the uers who are defaulters are the new users.
# 2) Old Users are trusted and they are mostly non defaulters.

# In[82]:


# Dropping the target value to fit the remaining data into standard scaler 
x2=df_clean.drop(['label'],axis=1)
x2
print(x2.shape)


# In[83]:


# Setting up the Target value in variable y1.
y1=df_clean['label']
y1.shape


# In[86]:


# .....................Importing Important libraries for Classification Models................
# Models from Scikit-Learn...
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Ensemble Techniques...
# from sklearn.ensemble import GradientBoostingClassifierx apviorn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier

# Model selection libraries...
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.model_selection import GridSearchCV

# Importing some metrics we can use to evaluate our model performance.... 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score


# In[93]:


# Function for GridSearch
from sklearn.model_selection import GridSearchCV
def grid_cv(mod,parameters,scoring):
    clf = GridSearchCV(mod,parameters,scoring, cv=5,verbose=1,n_jobs=-1,refit=True)
    clf.fit(x_train,y_train)
    print(clf.best_params_)


# # Conclusion:
# 
# 1) 28% of Users having negative or zero balance are defaulters, which is very high.
# 
# 2) 10% to 12% Users are defaulters which falls in the category of Average and Low balance category.
# 
# 3) Users having high balance and are defaulters are very less in number.
# 
# 4) Users who take more number of loans are non defaulters(i.e 98% of the category) as they repays the loan within the given time i.e 5 days.
# 
# 5) 14% of the Users are are among the average number of loan taken category are defaulters.
# 
# 6) 40 % of the Users who do not even recharged in the 90 days are defaulters only.
# 
# 7) Users who do very high amount of recharge always pays their loans on time. i.e 98% of them are non-defaulters.
# 
# 8) 34% of the Users who do less amount of recharge are defaulters.
# 
# 9) Users who did not take any loans are non defaulters.
# 
# 10) Most of the Users(i.e 97%) who take large amount of loans comes under non defaulter category.
# 
# 11) 17% of the users who take small loans are defaulters.
# 
# 12) Among the Users who have not done a single recharge in 3 months 40% are defaulters.
# 
# 13) Among the Users who are very frequent in recharging and who always pay their loans on time are more in number i.e 99% of the total category, which is a good news for the company.
# 
# 14) 32% of the uers who are defaulters are the new users.
# 
# 15) Old Users are trusted and they are mostly non defaulters.

# **Steps Followed:**
# 
# 1) Data Analysis.
# 
# 2) EDA Analysis.
# 
# 3) Different models are used and machine is trained for each models to find Best Accuracy Score.
# 
# 4) Best parameters are found using Gridsearch cv and applied to the best models.
# 
# 5) AUC ROC Curves are made for each model.
# 
# 6) A Result table is made comprises of accuracy,cross_val,auc_roc scores of each model.

# # End of the Document 

# In[ ]:




