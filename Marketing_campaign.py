import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


#Explore Dataset
df = pd.read_csv("C:/Users/Sunshine/Desktop/CEM/Continued_Education/Python/Data/marketing_campaign.csv")
pd.set_option('display.max_columns', None)
print(df.head(10))
print(df.info())
print(df.describe())

#Data cleaning and editing
#edit column names
df.rename(columns={'MntWines':'Wine', 
                   'MntFruits':'Fruit', 
                   'MntMeatProducts':'Meat', 
                   'MntFishProducts':'Fish', 
                   'MntSweetProducts':'Sweets', 
                   'MntGoldProds':'Gold', 
                   'NumWebPurchases':'Web_Purchases', 
                   'NumCatalogPurchases':'Catalog_Purchases', 
                   'NumStorePurchases':'Store_Purchases', 
                   'NumWebVisitsMonth':'WebVisits_perMonth',
                   'NumDealsPurchases':'Discount_Purchases',
                   'AcceptedCmp1' : 'First_Promotion', 
                   'AcceptedCmp2' : 'Second_Promotion', 
                   'AcceptedCmp3' : 'Third_Promotion', 
                   'AcceptedCmp4' : 'Fourth_Promotion',
                   'AcceptedCmp5' : 'Fifth_Promotion', 
                   'Response' : 'Last_Promotion'}, inplace=True)

#combine all purchase columns
df['Spent'] = df['Wine'] + df['Fruit'] + df['Meat'] + df['Fish'] + df['Sweets'] + df['Gold']
#drop the unnecessary columns
df.drop(['Z_CostContact','Z_Revenue'], axis = 1, inplace = True)

#explore null values
print('columns after edits')
print(df.columns)

##Visualizations
#Customer Profiling - who are the customers? 

#Marital Status just using pandas histogram
df['Marital_Status'].hist()
plt.xticks(rotation=45)
plt.xlabel('Marital Status')
plt.ylabel('# of Observations')
plt.title('Marital Status of Customers')
plt.show()

#Age using matplotlib.pyplot histogram
df['Age'] = 2021 - df['Year_Birth']
plt.hist(df['Age'], bins=20)
plt.xlabel('Age (years)')
plt.ylabel('# of Observations')
plt.title('Customer Ages')
plt.show()
#just checking to see if these Ages are valid
print(df[df['Age'] >100])
#create a binned age column
df['Ages'] = pd.cut(x=df['Age'], bins=[10, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129], labels = ['10s','20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s', '110s', '120s']) 

#Education using seaborn histogram
sns.histplot(df['Education'], color='g')
plt.xticks(rotation=45)
plt.show()

#kids using subplots
fig0, ax0 = plt.subplots(1,2, sharey=True)
ax0[0].hist(df['Kidhome'])
ax0[1].hist(df['Teenhome'])
ax0[0].set_xlabel('Number of Kids')
ax0[0].set_ylabel('Count')
ax0[1].set_xlabel('Number of Teens')
plt.subplots_adjust( wspace=0.3)
plt.show()
#Combining the kids to one dataset
df['Children_Home']= df['Kidhome'] + df['Teenhome']
plt.hist(df['Children_Home'], color='r')
plt.xticks(rotation=45)
plt.xlabel('# of Children')
plt.ylabel('# of Observations')
plt.title('Number of Children')
plt.show()


#Who is spending the most? total and average
#total
fig1, ax1 = plt.subplots(2,2, sharey=True)
ax1[0,0].bar(['Graduation', 'PhD', 'Master', 'Basic', '2nd Cycle'], df.groupby('Education')['Spent'].sum())
ax1[0,1].bar(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'], df.groupby('Ages')['Spent'].sum())
ax1[1,1].bar(['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'Yolo'], df.groupby('Marital_Status')['Spent'].sum())
ax1[1,0].bar([0,1,2,3], df.groupby('Children_Home')['Spent'].sum())
ax1[0,0].set_xticklabels(labels=['Graduation', 'PhD', 'Master', 'Basic', '2nd Cycle'], rotation=45)
ax1[0,0].set_ylabel('Dollars Spent')
ax1[0,0].set_xlabel('Education')
ax1[0,1].set_xticklabels(labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'], rotation=45)
ax1[0,1].set_xlabel('Age')
ax1[1,1].set_xticklabels(labels=['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'Yolo'], rotation=45)
ax1[1,1].set_xlabel('Marital_Status')
ax1[1,0].set_xticklabels(labels=[0,1,2,3], rotation=45)
ax1[1,0].set_ylabel('Dollars Spent')
ax1[1,0].set_xlabel('Number of Kids')
plt.subplots_adjust( hspace=1.1)
plt.show()
#average
sns.catplot(x='Education', y='Spent', data=df, kind='bar')
plt.xticks(rotation=45)
plt.show()
sns.catplot(x='Marital_Status', y='Spent', data=df, kind='bar')
plt.xticks(rotation=45)
plt.show()
sns.catplot(x='Children_Home', y='Spent', data=df, kind='bar')
plt.xticks(rotation=45)
plt.show()

#Does income correlate to how much a customer spends
sns.scatterplot(x='Income',y='Spent',color='g',data=df)
plt.show()
#let's find the trend 
sns.lmplot(x='Income',y='Spent', data=df, truncate=True)
plt.show()
#remove outliers
q_low = df["Income"].quantile(0.01)
q_hi  = df["Income"].quantile(0.99)
df_filtered = df[(df["Income"] < q_hi) & (df["Income"] > q_low)]
sns.lmplot(x='Income',y='Spent', data=df_filtered, truncate=True, logistic=True)
plt.show()
#log regression looks like it would fit better so lets try ski-kit learn


#Web vs Catalog vs Store vs web purchase. Where are people buying the merch?
Purchases_location = [df['Web_Purchases'].sum(), df['Catalog_Purchases'].sum(), df['Store_Purchases'].sum()]
Purchases_loc_columns = ['Web Purchases','Catalog Purchases','Store Purchases']
plt.bar(Purchases_loc_columns, Purchases_location)
plt.xlabel('Purchase Location')
plt.ylabel('Number of Purchases')
plt.show()


#Which products are being bought?
Products_bought = [df['Fish'].sum(), df['Wine'].sum(), df['Meat'].sum(), df['Sweets'].sum(), df['Gold'].sum(), df['Fruit'].sum()]
Product_name_columns = ['Fish','Wine','Meat', 'Sweets', 'Gold', 'Fruit']
plt.bar(Product_name_columns, Products_bought)
plt.xlabel('Products')
plt.ylabel('Number of Purchases')
plt.show()

#Which products are being bought?
Products_bought = [df['Fish'].sum(), df['Wine'].sum(), df['Meat'].sum(), df['Sweets'].sum(), df['Gold'].sum(), df['Fruit'].sum()]
Product_name_columns = ['Fish','Wine','Meat', 'Sweets', 'Gold', 'Fruit']
plt.bar(Product_name_columns, Products_bought)
plt.xlabel('Products')
plt.ylabel('Number of Purchases')
plt.show()


#Did the campaigns work? if so which ones?
Promotion_Totals = [df['First_Promotion'].sum(), df['Second_Promotion'].sum(), df['Third_Promotion'].sum(), df['Fourth_Promotion'].sum(), df['Fifth_Promotion'].sum(), df['Last_Promotion'].sum()]
Promotion_name_columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Last']
plt.bar(Promotion_name_columns, Promotion_Totals)
plt.xticks(rotation=45)
plt.xlabel('Promotion')
plt.ylabel('Number of Purchases')
plt.show()


#does WebVisits_perMonth correlate to the #purchases
sns.regplot(x='WebVisits_perMonth', y='Web_Purchases', data=df)
plt.xlabel('Web Visits per Month')
plt.ylabel('Web Purchases')
plt.show()
#looks like we need to get rid of some outliers.... 


#Who complains
df[['Age', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'Income','Complain']].head(20)
Complaints = df.groupby('Age')['Complain'].sum()
print(Complaints)
#need to group ages by decades.. 
df['Ages'] = pd.cut(x=df['Age'], bins=[10, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129], labels = ['10s','20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s', '110s', '120s']) 
print(df['Ages'])
Complaints_bin = df.groupby('Ages')['Complain'].sum()
print(Complaints_bin)
Complaints_bin.plot(kind='bar', rot=45, title='Complaints by Age')
plt.ylabel('Number of Complaints')
plt.show()


print(df['Income'].max())
print(df['Income'].min())
df['Incomes'] = pd.cut(x=df['Income'], bins=[0, 49999, 99999, 499999, ], labels = ['<50K', '<100K', '<500K']) 
Complaints = df.groupby('Incomes')['Complain'].sum()
Complaints.plot(kind='bar', rot=45, title='Complaints by Income') 
plt.ylabel('Number of Complaints')
plt.show()
print(Complaints)


df_corr = df[['Fish','Wine','Meat', 'Sweets', 'Gold', 'Fruit','Age', 'Education', 'Marital_Status', 'Children_Home', 'Income','Complain']]
corr = df_corr.corr()
sns.heatmap(corr)
corr = df.corr()
sns.heatmap(corr)
