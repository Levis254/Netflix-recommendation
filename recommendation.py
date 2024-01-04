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
    text=re.sub('\[.*?\]', '', text)
    text=re.sub('https?://\S+|www.\.\S+', '', text)
    text=re.sub('<.*?>+', '', text)
    text=re.sub('[%s]'% re.escape(string.punctuation), '', text)
    text=re.sub("\n",'', text)
    text=re.sub('\w*\d\w*',"",text)
    text=[word for word in text.split('') if word not in stopword]
    text="".join(text)
    return text

data["Title"]=data["Title"].apply(clean)


#view the cleaned title column

print(data.Title.sample(10))

#using the genres column, find the similarity in content (using cosine similarity)

#first convert the Genre table to a list
feature=data["Genre"].tolist()

tfidf=text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix=tfidf.fit_transform(feature)
similarity=cosine_similarity(tfidf_matrix)


#set the title column as an index so that we can find similar content
#by giving the title of the movies or TV show as an input

indices=pd.Series(data.index, index=data['Title']).drop_duplicates()

#function to recommend Movies and TV shows

def netflix_recommendation(title, similarity=similarity):
    index=indices[title]
    similarity_scores=list(enumerate(similarity[index]))
    similarity_scores=sorted(similarity_scores, key=lambda x: x[1])
    similarity_scores=similarity_scores[0:10]
    movieindices=[i[0] for i in similarity_scores]
    return data['Title'].iloc[movieindices]

print(netflix_recommendation("girlfriend"))