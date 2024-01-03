#import necessary python libraries

import numpy as np
import pandas as pd 
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

#read the data
data=pd.read_csv("C:/Users/levis/Downloads/archive/netflixData.csv")

#print the first five rows of the tabulated data
print(data.head())

#calculate the total null values in each column feature

print(data.isnull().sum())

#next we select the columns we will use to build the recommendation system

data=data[["Title", "Description", "Content Type", "Genres"]]

print(data.head())


#drop rows wil null values
data=data.dropna()

#Clean the Title columns since it requires some data preparation

import nltk
import re
nltk.download('stopwords')

stemmer=nltk.SnowballStemmer("english")

from nltk.corpus import stopwords

import string

stopword=set(stopwords.words('english'))

def clean(text):
    text=str(text).lower()