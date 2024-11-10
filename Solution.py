#!/usr/bin/env python
# coding: utf-8

# # Part 1
# # EDA & Data Preprocessing on Google App Store Rating Dataset.

# ## Q1. Import required libraries and read the dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df1 = pd.read_csv(r"C:\Users\USER\Desktop\New folder (2)\Apps_data+(1).csv")


# ## Q2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features.

# In[3]:


# Let us look at the first 10 sample data
df1.head(10)


# In[4]:


# Shape of the data
df1.shape


# In[5]:


# Basic information about dataset and it's datatypes
df1.info()


# ## Q3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model building.

# In[6]:


# Basic statistical data of numerical columns
df1.describe().round(2)


# In[7]:


# Basic statistical data of numerical and categorical columns
df1.describe(include = 'all').round(2)

The columns that needs some changes are:
1. Rating
2. Reviews
3. Size
4. Installs
# ## Q4. Check if there are any duplicate records in the dataset? if any drop them.

# In[8]:


# Checking the number of duplicate rows
print('The number of duplicate rows are:')
df1.duplicated().sum()


# In[9]:


# Dropping the duplicate rows permanently
df1.drop_duplicates(inplace=True)


# In[10]:


# Checking if they are dropped
df1.duplicated().sum()


# In[11]:


df1.shape


# ## Q5. Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them.

# In[12]:


# Looking at the unique entries of 'Category' column
list(df1['Category'].unique())


# In[13]:


# Removing the category that might be invalid for our analysis
df1 = df1[df1['Category']!='1.9']


# In[14]:


# After removal of unwanted entry
df1['Category'].unique()


# ## Q6. Check if there are missing values present in the column Rating, If any? drop them and and create a new column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)

# In[15]:


print('No of missing values in Rating column:',df1['Rating'].isna().sum())


# In[16]:


# Drop the missing values
df1['Rating'].isnull().sum()


# In[17]:


df1 = df1.dropna(subset='Rating')


# In[18]:


# New column 'Rating_Category' as 'High' for rating > 3.5 and others as 'Low'
df1['Rating_Category'] = df1['Rating'].map(lambda x : 'High' if x>3.5 else 'Low')


# ## Q7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution.

# In[19]:


df1['Rating_Category'].value_counts()


# In[20]:


sns.countplot(x = df1['Rating_Category'])
plt.show()

There are 8012 apps with 'High' reviews and 880 apps with 'Low' reviews.
# ## Q8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and handle the outliers using a transformation approach.(Hint: Use log transformation)

# In[21]:


# Converting 'Reviews' column to integer type data
df1['Reviews'] = df1['Reviews'].astype('int64')


# In[22]:


# Boxplot to check for outliers
sns.boxplot(df1['Reviews'])
plt.show()

There seems to be considerably high number of outliers and needs to be suitably treated.We need to drop all the zero values from the 'Reviews' column before applying log transformation since log(0) is undefined
# In[23]:


df1 = df1[df1['Reviews']!=0]


# In[24]:


# Log transformation to treat outliers
log = np.log(df1['Reviews'])


# In[25]:


log


# In[26]:


# Boxplot after log transformation
sns.boxplot(log)
plt.show()

Outliers have been effectively treated through log transformation and all data have been narrowed down to a certain range.
# ## Q9. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into suitable data type.

# In[27]:


# Replacing 'M' with six zeros and 'k' with three zeros
df1['Size'] = df1['Size'].str.replace('M', '000000')
df1['Size'] = df1['Size'].str.replace('k', '000')


# In[28]:


df1['Size'].head(10)


# In[29]:


df1 = df1[df1['Size']!= 'Varies with device']


# In[30]:


df1['Size'].astype('float64')


# In[31]:


df1.reset_index(drop=True)


# ## Q10. Check the column 'Installs', treat the unwanted characters and convert the column into a suitable data type.

# In[32]:


df1['Installs']


# In[33]:


# '+' sign and comma(,) to be removed
df1['Installs'] = pd.to_numeric(df1['Installs'].str.replace('[+,]', '', regex= True))


# In[34]:


df1[['Installs']]


# In[35]:


# Checking the datatype
df1['Installs'].dtype


# ## Q11. Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type.

# In[36]:


df1['Price'].dtype


# In[37]:


df1['Price'] = pd.to_numeric(df1['Price'].str.replace('$', ''))


# In[38]:


df1['Price'].dtype


# In[39]:


df1['Price'].sort_values(ascending=False)


# ## Q12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver' columns since which are redundant for our analysis)

# In[40]:


columns_drop = ['Rating', 'App', 'Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver']
df1.drop(columns = columns_drop, inplace= True)


# In[41]:


df1


# ## Q13. Encode the categorical columns.

# In[42]:


# Get info of all columns to see categorical columns
df1.info()

We find four columns that can be encoded
1. Category
2. Type
3. Content Rating
4. Rating_Category
# ### Let us use Label Encoding - It assigns a unique number to each category of a column to make it easier for the machine to understand the data.

# In[43]:


# Import the necessary libraries
from sklearn.preprocessing import LabelEncoder


# In[44]:


labelencoder = LabelEncoder()


# ##### 1. Category

# In[45]:


# Check for the categories of the column 'Category'
print(df1['Category'].unique())


# In[46]:


print('No of unique categories :',df1['Category'].nunique())


# In[47]:


# Transform 'Category' using label encoding
df1['Category'] = labelencoder.fit_transform(df1['Category'])


# In[48]:


# After label encoding
df1['Category'].value_counts()


# ### 2. Type

# In[49]:


df1['Type'].unique()


# In[50]:


df1['Type'] = labelencoder.fit_transform(df1['Type'])


# In[51]:


df1.head()


# In[52]:


df1['Type'].value_counts()


# ### 3. Content Rating

# In[53]:


# Check for the categories of the column 'Content Rating'
df1['Content Rating'].unique()


# In[54]:


df1['Content Rating'].nunique()


# In[55]:


df1['Content Rating'] = labelencoder.fit_transform(df1['Content Rating'])


# In[56]:


df1.head()


# In[57]:


df1['Content Rating'].value_counts()


# ### 4. Rating Category

# In[58]:


df1['Rating_Category'].unique()


# In[59]:


df1['Rating_Category'] = labelencoder.fit_transform(df1['Rating_Category'])


# In[60]:


df1.head()


# In[61]:


df1['Rating_Category'].value_counts()


# ## Q14. Segregate the target and independent features (Hint: Use Rating_category as the target)

# In[62]:


# We have to use 'Rating_Category' as the target

X = df1.drop('Rating_Category', axis = 1)  # Independent variable

Y = df1[['Rating_Category']]   # Dependent variable


# In[63]:


X


# In[64]:


Y


# ## Q15. Split the dataset into train and test.

# In[65]:


# Import the necesary libraries
from sklearn.model_selection import train_test_split


# In[66]:


# Let us split the model into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.20, random_state=10)


# In[67]:


X_train


# In[68]:


X_test


# In[69]:


Y_train


# In[70]:


Y_test


# ## Q16. Standardize the data, so that the values are within a particular range.

# In[71]:


#Importing necessary libraries.
from sklearn.preprocessing import StandardScaler


# In[72]:


scaler = StandardScaler()


# In[73]:


df1.info()


# In[74]:


# Standardizing 
df1 = scaler.fit_transform(df1)


# In[75]:


pd.DataFrame(df1)


# # End of Part 1

# # PART II: Data Visualization on Honey Production dataset using seaborn and matplotlib libraries.

# ## Q1. Import required libraries and read the dataset.

# In[76]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[77]:


df2 = pd.read_csv(r"C:\Users\USER\Desktop\New folder (2)\honeyproduction.csv")


# ## Q2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features.

# In[78]:


df2.head()


# In[79]:


df2.shape


# In[80]:


df2.info()


# In[81]:


df2.isnull().sum()


# ## Q3. Display the percentage distribution of the data in each year using the pie chart.

# In[82]:


pie_year = df2['year'].value_counts()


# In[83]:


plt.figure(dpi=150)
plt.pie(pie_year, autopct= '%.1f%%', labels= pie_year.index, pctdistance = 0.8)
plt.show()


# ## Q4. Plot and Understand the distribution of the variable "price per lb" using displot, and write your findings.

# In[84]:


plt.figure(figsize=(7,5), dpi=300)
sns.displot(x = df2['priceperlb'], kde= True)
plt.show()

The data is right-tailed which means it is positively skewed.
Mean > Median
# ## Q5. Plot and understand the relationship between the variables 'numcol' and 'prodval' through scatterplot, and write your findings.

# In[85]:


plt.figure(figsize=(5,3), dpi=150)
sns.scatterplot(x = df2['numcol'], y = df2['prodvalue'])
plt.show()

The data points are always rising upwards. Hence, we can say that 'numcol' and 'prodvalue' are directly proportional.
The number of honey producing colonies has a direct impact on the production value.
# ## Q6. Plot and understand the relationship between categorical variable 'year' and a numerical variable 'prodvalue' through boxplot, and write your findings.

# In[86]:


plt.figure(figsize=(9,7), dpi=150)
sns.boxplot(x = df2['year'], y = df2['prodvalue'])
plt.xticks(rotation  = 45)
plt.show()

The above boxplot gives us information about the outliers of the 'prodvalue' column for each year.
As we can see that there are potentially good amount of outliers and the range of the outlier is increasing every year.
This shows that the production value has genrally increased as the years have passed by.
The trend goes up till year 2003 and has seen a minor drop for the next 3 years.
The highest value was reached in the year 2010.
# ## Q7. Visualize and understand the relationship between the multiple pairs of variables throughout different years using pairplot and add your inferences. (use columns 'numcol', 'yield percol', 'total prod', 'prodvalue','year')

# In[87]:


columns = ['numcol', 'yieldpercol', 'totalprod', 'prodvalue','year']
sns.pairplot(df2[columns], hue = 'year')
plt.show()


# ## Q8. Display the correlation values using a plot and add your inferences. (use columns 'numcol', 'yield percol', 'total prod', 'stocks', 'price per lb', 'prodvalue')

# In[88]:


plt.figure(figsize=(5,3), dpi=300)
columns = ['numcol', 'yieldpercol', 'totalprod', 'stocks', 'priceperlb', 'prodvalue']
corr_matrix  = df2[columns].corr()
sns.heatmap(corr_matrix, annot= True, cmap = 'viridis')

1.'numcol' and 'prodvalue' have the highest correlation co-efficent indicating that the two share a very strong positive linear relationship. The same is the relation between 'numcol' and 'totalprod' since 'prodvalue' is obtained through 'totalprod'.

2. 'yieldpercol' and 'priceperlb' have the lowest correlation coefficient of -0.36. Negative sign indicates that as one parameter increases the other decreases. Here, as the yield of honey per colony increases, the price per pound can be decreased.
If the yield is low, the prices tend to be high.
# # End of Part 2
# # End of Project
