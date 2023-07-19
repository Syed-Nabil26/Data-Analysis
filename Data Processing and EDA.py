#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")

import seaborn as sns


import ast




import warnings
warnings.filterwarnings("ignore")


# 

# # Data Understanding

# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


movies = pd.read_csv("movies_metadata.csv")
movies.head(3)


# In[4]:


movies.shape


# In[5]:


movies.info()


# In[6]:


movies.isna().sum()


# In[7]:


movies.describe()


# #### Concluding some points after reviewing the Data Sets
# 
# * Inappropiate data types are assigned
# * Missing values are close to 90% for columns like "belongs_to_collection" ; "homepage" ; "tagline"
# * Need to drop some columns as they're not going to help us during the Analysis['homepage','original_title','tagline','spoken_languages','overview','poster_path','video']

# # Data Processing

# #### Dropping columns

# In[8]:


movies = movies.drop(movies[['homepage','original_title','tagline','spoken_languages','overview','poster_path','video']] , axis = 1 )


# In[9]:


movies.head(2)


# #### Dropping Null values from the Entire Data Frame i.e If Any row contains entire Null Values

# In[10]:


movies = movies.dropna(how = "all" )


# #### Converting into Correct Data Types

# In[11]:


movies['id'] = pd.to_numeric(movies['id'],errors='coerce',downcast='integer')
movies['budget'] = pd.to_numeric(movies['budget'],errors='coerce',downcast='float')
movies['popularity'] = pd.to_numeric(movies['popularity'],errors='coerce',downcast='float')


# #### Converting the Release date  column into date time and removing Null values 

# In[12]:


movies['release_date'] = pd.to_datetime(movies['release_date'],errors='coerce')
movies.dropna(subset = ['release_date'],axis=0,inplace=True)


# #### Creating New Columns Release year and Release Month from "Release_Date" Column after dropping Null Values

# In[13]:


movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month


# #### The Above Value is from Json Script stringified and Store under the Dictionary Format and upon that List and it contains more then 40k Null values so we are filling Null values with "None" and converting it into Integer

# In[14]:


movies['belongs_to_collection'] = movies['belongs_to_collection'].fillna('None')
movies[['collection_name']] = movies[['belongs_to_collection']].applymap(lambda x: 'None' if x=='None' else ast.literal_eval(x)['name'])
movies['belongs_to_collection'] = movies['belongs_to_collection'].map(lambda x: 0 if x == 'None' else 1)


# In[15]:


day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
def get_day(x):
    try:  
        answer = x.weekday()
        return day_order[answer]
    except:
        return np.nan

movies['day'] = movies['release_date'].apply(get_day)


# In[16]:


movies.isna().sum()


# #### Treating other Missing Columns In DataSet 
# * Status Column Missing value is replace with the Mode
# * Since Run time consist of 0 minutes need to be treated and replaced with Average Running time

# In[17]:


movies["status"].value_counts()


# In[18]:


movies['runtime'].value_counts()


# In[19]:


movies['status'].fillna(movies['status'].mode()[0],inplace=True)
movies['runtime'] = movies['runtime'].replace(0,np.nan)
movies['runtime'].fillna(movies['runtime'].mean(),inplace=True)
movies['original_language'].fillna(movies['original_language'].mode()[0],inplace=True)


# In[20]:


movies.isna().sum()


# In[21]:


movies.imdb_id.dropna()


# #### Creating column to see Movie is Hit / Flop

# In[22]:


movies['return'] = movies['revenue'] / movies['budget']
movies['success_or_flop'] = movies['return'].map(lambda x: 1 if x>=1 else 0)
movies.head()


# In[23]:


movies.reset_index(inplace=True,drop=True)
movies = movies.dropna(subset='imdb_id')
movies.reset_index(inplace=True,drop=True)


# #### The Above Value is from Json Script stringified and Store under the Dictionary Format and upon that List and it contains To extract values from it we are using this function by importing ast(abstract syntax tree ) and Literal eval to review data type for Machine. Using Isinstance it will return ans in boolean format to check our passed obj and parameter is same or not 

# In[24]:


def json_to_list(row,want='name'):
    new_list = ast.literal_eval(row)    
    if new_list == [] or isinstance(new_list,float):
        return (np.nan)
    else:
        inner = []
        for j in new_list:
            inner.append(j[want])
        return (inner)
    
movies[['production_companies']] = movies[['production_companies']].applymap(json_to_list)
movies[['production_countries']] = movies[['production_countries']].applymap(lambda row: json_to_list(row ,'iso_3166_1'))
movies[['genres']] = movies[['genres']].applymap(json_to_list)


# In[25]:


movies


# #### Most of the Values of Budget and Revenue is 0 
# * Its Better to replace them with np.nan 

# In[26]:


movies[movies['budget'] == 0].shape , movies[movies['revenue'] == 0].shape


# In[27]:


movies['budget'].replace(0,np.nan , inplace = True)
movies['revenue'].replace(0,np.nan , inplace = True)


# In[28]:


movies.isna().sum()


# #### Its observed that the values is not scaled in Budget and Revenue i.e in place of 1 Million its written 1 and same like that for many columns in the data set so replacing them with the logic 

# In[29]:


def scale_money(num):
    if num < 100:
        return num * 1000000
    elif num >= 100 and num < 1000:
        return num * 10000
    elif num >= 1000 and num < 10000:
        return num *100
    else:
        return num
    
movies[['budget', 'revenue']] = movies[['budget', 'revenue']].applymap(scale_money)


# #### A function to extract unique values from the column except np.nan and the values are going to be in a dictionary format

# In[30]:


def get_all_items(df , col):
    all_items = {}
    for row in df[col]:
        counter = 0
        if row == np.nan or isinstance(row,float) :
            continue
        for single_value in row:  
            value = all_items.get(single_value)
            if value == None:
                all_items[single_value] = counter + 1
            else:
                all_items[single_value] = value + 1
    return all_items


# #### Extracting the Genres

# In[31]:


all_genres = get_all_items(movies , 'genres')


# In[32]:


all_genres


# #### Extracting companies

# In[33]:


all_companies = get_all_items(movies ,'production_companies')
all_companies


# #### Extracting - Country Names

# In[34]:


major_prod_company = {k:v for (k,v) in all_companies.items() if v > 50}

all_countries = get_all_items(movies ,'production_countries')


# In[35]:


movies.columns


# #### Using Group By Function on top of Categorical data i.e Collection name to check the status of Revenue , Budget and Title on top of the aggregate functions like mean ,sum , count and these all are passed in a Dictionary Format and to find out more than Two aggregate functions of a field we are passing it in list

# In[36]:


franchise = movies.dropna().groupby(by='collection_name').agg({'revenue':['sum','mean'],'budget':['sum','mean'],'title':'count','popularity':'mean'})
franchise


# In[37]:


most_films_franchise = franchise.sort_values([('title','count')],ascending=False)[1:21]


# In[38]:


most_films_franchise.shape


# # Exploratory Data Analysis

# #### Plotting Bar graph Against Budget vs Franchise

# In[ ]:





# In[43]:


plt.figure(figsize=(20,5))
ax = sns.barplot(y=most_films_franchise[('revenue','sum')],x=most_films_franchise.index, palette = sns.color_palette("YlOrBr"))
plt.xlabel('Franchises')
plt.title('Franchises with their total gross earnings')
plt.ylabel('Gross Earnings ')
plt.xticks(rotation=90)


count = most_films_franchise[('title','count')]
ax.bar_label(ax.containers[0], labels=count, padding=3)

plt.show()


# #### From the above visuals we can find out the Top Highest Grossing Movies Franchise 

# In[ ]:





# #### -----------------------------------------------------------------

# In[ ]:





# In[46]:


franchise_mean_budget = franchise.sort_values([('budget','mean')],ascending=False)[1:20]

plt.figure(figsize=(20,5))
ax = sns.barplot(y=most_films_franchise[('budget','mean')],x=most_films_franchise.index, palette = sns.color_palette("YlOrBr"))
plt.xlabel('Franchises')
plt.title('Franchises with their Budget')
plt.ylabel('Avg - budget')
plt.xticks(rotation=90)




plt.show()


# #### From above visuals we can findout the Most Expensive Franchises

# #### --------------------------------------------------------------------------------------

# In[59]:


Most_Expensive = franchise.sort_values([('budget','sum')],ascending=False)[1:11]
Avg_Expensive = franchise.sort_values([('budget','mean')],ascending=False)[1:11]


# In[58]:


HG = franchise.sort_values([('revenue','sum')],ascending=False)[1:11]
Avg_HG= franchise.sort_values([('revenue','mean')],ascending=False)[1:11]


# In[65]:


plt.figure(figsize=(20,10))
ax = sns.barplot(y=Most_Expensive[('budget','mean')],x =Most_Expensive.index, palette = sns.color_palette("YlOrBr"))
plt.xlabel('Series')
plt.title('Most Expensive')
plt.ylabel('Budget')
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# #### Since we have extracted the values by using Function(get_all) and its in dictionary format so we are using Keys and Values out of it

# In[72]:


fig = plt.figure(figsize = (20, 10))
all_genre_keys = list(all_genres.keys())
all_genre_values = [int(i) for i in all_genres.values()]
sns.barplot(x=all_genre_keys, y = all_genre_values)
plt.xticks(rotation=90)
plt.xlabel('Genres')
plt.ylabel('Movie Counts')
plt.title('Total Movies in All Categories')
plt.show()


# #### Top Genres of Movies
# 

# In[ ]:





# #### Filling the Null Values with top 3 Categories in a random way

# In[76]:


import random


# In[77]:


choices = [['Drama'],['Comedy'],['Thriller']]
movies['genres'] = [random.choice(choices) if isinstance(x,float) else x for x in movies['genres']]


# In[79]:


movies.genres.isna().sum()


# #### Major Production Companies

# In[114]:


fig = plt.figure(figsize = (20, 10))
prod_keys = list(major_prod_company.keys())[0:15]
prod_values = [int(i) for i in major_prod_company.values()][0:15]
sns.barplot(x=prod_keys, y = prod_values , palette = sns.cubehelix_palette(start=.5, rot=-.5))
plt.xticks(rotation=90)
plt.xlabel('Major Production Companies')
plt.ylabel('No. of movies ')
plt.title('Production companies of movies')
plt.show()


# #### Movie Release Each year

# In[111]:


year_name = movies['release_year'].value_counts().index.tolist()
year_count = movies['release_year'].value_counts().tolist()


# In[110]:




fig = plt.figure(figsize = (25, 15))
sns.barplot(x=year_name,y=year_count)
plt.xticks(rotation=90)
plt.xlabel('Release Year')
plt.ylabel('No. of movies released')
plt.title('Movies Released each year')
plt.show()


# ####  Top Languages

# In[112]:


lang_name = movies['original_language'].value_counts().index.tolist()[0:6]
lang_count = movies['original_language'].value_counts().tolist()[0:6]

fig = plt.figure(figsize = (10,8))
sns.barplot(x=lang_name,y=lang_count)
plt.xticks(rotation=45)
plt.xlabel('Languages')
plt.ylabel('No. of movies')
plt.title('Languages in which movies released')
plt.show()


# In[116]:


fig = plt.figure(figsize = (25, 5))
countries_keys = list(all_countries.keys())[0:15]
countries_values = [int(i) for i in all_countries.values()][0:15]
sns.barplot(x=countries_keys, y = countries_values)
plt.xticks(rotation=45)
plt.xlabel('Major Production Countries')
plt.ylabel('No. of movies produced ')
plt.title('Production countries of movies')
plt.show()


# #### # Filling NaN values in 'production_countries' with major production countries
# 

# In[115]:


choices = [['US','DE'],['GB','FR']]
movies['production_countries'] = [random.choice(choices) if isinstance(x,float) else x for x in movies['production_countries']]


# ### Revenue of Movies Each Year

# In[122]:


plt.figure(figsize=(30,10))
sns.barplot(data = movies, x='release_year',y='revenue')
plt.xticks(rotation=90)
plt.show()


# ###  Movies Runtime

# In[124]:


fig = plt.figure(figsize = (30, 10))
sns.lineplot(data= movies , x= 'release_year',y='runtime')
plt.ylabel('runtime(minutes)')
plt.xticks(np.arange(1874, 2024, 5.0))
plt.title('Change in movies runtime with years')
plt.show()


# ### Average Vote Count

# In[126]:


plt.figure(figsize=(30,10))
sns.barplot(x=movies['release_year'],y=movies['vote_count'], palette = sns.color_palette("dark:salmon_r"))
plt.xticks(rotation=90)
plt.show()


# In[128]:


largest_runtime = movies.nlargest(10,'runtime')[['runtime','release_year','title']]

plt.figure(figsize=(20,5))
sns.barplot(x=largest_runtime['runtime'],y=largest_runtime['title'], palette = sns.color_palette("Blues"))
plt.xlabel('runtime(minutes)')
plt.title('Top 10 Largest runtime movies ')
plt.show()


# In[129]:


plt.figure(figsize=(20,5))
sns.countplot(x='day',data=movies,palette = sns.color_palette("coolwarm"))
plt.ylabel('No. of movies released')
plt.title('No of movies released on each weekday')
plt.show()


# In[130]:


plt.figure(figsize=(20,5))
sns.countplot(x='release_month',data=movies,palette = sns.color_palette("coolwarm"))
plt.ylabel('No. of movies released')
plt.title('No of movies released on each month')
plt.show()


# In[132]:


plt.figure(figsize=(20,5))
sns.barplot(x='release_month',y='revenue',data=movies,palette = sns.color_palette("coolwarm"))
plt.ylabel('No. of movies released')
plt.title('Months in which movies made highest revenue')
plt.show()


# ### There are other data set which can be used for further analysis process

# In[ ]:




