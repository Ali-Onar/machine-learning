# -*- coding: utf-8 -*-

import pandas as pd

#%% import twitter data

# Veri setinde twitterdan çekilen tweetler ve tweetleri atanların cinsiyetleri yer almakta
# Bizde tweetlere bakarak kadın mı yazmış erkek mi yazmız, text classification yaparak öğrenmeye çalışacağız.
data = pd.read_csv(r"gender_classifier.csv", encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1) #concat: iki tane series veya dataframe'i birleştir
data.dropna(axis = 0, inplace = True) #nan değerleri yok et, data'ya eşitle

# Sınıflandırma algoritmalarında labelleri integer veya kategorical yapmamız gerekiyor. O yüzden gender'i integer'a çevirelim.
data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% Cleaning Data

# Regular Expression: RE ile veri setinden alfabede olmayan ifadeleri çıkaracağız
import re

#herhangi bir sample'ı seçtim
first_desc = data.description[30]

# ^ ile alfade dışındaki ifadeleri seç, yerine "boşluk" yap, bunu first_desc'e uygula
desc = re.sub("[^a-zA-Z]", " ", first_desc)

# preprocess / büyük harfleri küçük harfe çevirme
desc = desc.lower()

#%% Irrelavant Words (Gereksiz Kelimeler)

import nltk #Natural Language Tool Kit
nltk.download("stopwords") #corpus klasörüne indiriliyor
nltk.download('punkt')
from nltk.corpus import stopwords #corpus klasöründen import ediyoruz

#%% split yerine tokenize kullandık çünkü split don't kelimesini do not olarak ayırmaz
#desc = desc.split()
desc = nltk.word_tokenize(desc)

#%% gereksiz kelimeleri çıkar
desc = [ word  for word in desc  if not word in set(stopwords.words("english"))]


#%% Lemmatization

import nltk as nlp

lemma = nlp.WordNetLemmatizer()
# disc içindeki tüm kelimeleri dolan hepsinin kökünü bul
desc = [lemma.lemmatize(word) for word in desc ]

# list halinde olan desc'i araya boşluk koyarak birleştir
desc = " ".join(desc)


#%% Data Cleaning

description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [ word  for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description ]
    description = " ".join(description)
    description_list.append(description)
    
#%% Sklearn ile NLP

from sklearn.feature_extraction.text import CountVectorizer #bag of words oluşturmak için kullanılılan metot

# algoritmayı yormamak için en çok kullanılan kelimeler arasından 500 tanesini seç
max_features = 5000
count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")

# sparce_matrix: yukarda anlattığımız tablolarda 1 ve 0'lardan oluşan matrise verilen ad
# description_list'i yukarda tanımladığımız şekilde modele uyarlayıp fit yaptık, bunuda sparce_matrix'e eşitledik
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() #input

print("En sık kullanılan {} kelime: {}".format(max_features, count_vectorizer.get_feature_names()))

#%% Text Classification

y = data.iloc[:,0].values # gender: kadın, erkek sınıfı
x = sparce_matrix

#train test split
from sklearn.model_selection import train_test_split

#Modelin %10'unu test için ayıralım.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

#%% naive bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

#%% prediction

y_pred = nb.predict(x_test)
print("accuracy: ", nb.score(y_pred.reshape(-1,1), y_test))

