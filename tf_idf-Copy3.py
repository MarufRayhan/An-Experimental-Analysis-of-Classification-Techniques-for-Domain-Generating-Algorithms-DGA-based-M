#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import pandas as pd
import numpy as np
import time

from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


# In[2]:


nltk.download('punkt')


# In[3]:


from nltk.util import ngrams
from nltk.tokenize import word_tokenize
tokens = nltk.word_tokenize("I eat rice")
print(tokens[0])
grams = ngrams(tokens,2)
print(list(grams))


# In[4]:


# from nltk.tokenize import word_tokenize
# from nltk.util import ngrams

# def get_ngrams(text, n ):
#     text = re.split(r"\W+",str(text))
#     joined_text = " ".join(text)
# #     n_grams = ngrams(word_tokenize(text), n)
# #     return [ ' '.join(grams) for grams in n_grams]
#     return joined_text
          
          
# from nltk.tokenize import word_tokenize
# from nltk.util import ngrams

def get_ngrams(text,n):
    text = re.split(r"\W+",str(text))
    print("original text",text)
#     text = text[0]
    text = " ".join(text)
    tokens = nltk.word_tokenize(text)
    joined_text = " ".join(tokens)
    
#     stemming = PorterStemmer()
#     joined_text = [stemming.stem(w) for w in joined_text]
    
    print("Joined text", joined_text)
    return joined_text
    
#     n_grams = ngrams(word_tokenize(text), n)
    

#     grams = ''
#     for gram in n_grams:
#         print(gram)
#         for g in gram:
#             print(g)
#             if len(grams) == 0:
#                 grams = g
#                 print("grams here",grams)
#             else:
#                 grams = grams + ',' + g
#                 print("haha",grams)
#     if len(grams) == 0:
#         grams = text
#     return grams


# In[5]:


# result = get_ngrams("www.facebook.com", 0)


# In[6]:


# result


# In[7]:


def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(get_ngrams(element,1))
    return cleaned_X


# In[8]:


my_file = pd.read_csv("/home/bjit/testing_work/final__csv.csv")
my_file = my_file.sample(frac=1)
print(my_file.head())


# In[9]:


my_file['URL'] = my_file['URL'].str.lower()


# In[10]:


my_file['URL'][1]


# In[11]:


my_file['clean_url'] = apply_cleaning_function_to_list(my_file['URL'])


# In[12]:


row_list = []
for file in my_file['clean_url']:
    row_list.append(file)


# In[13]:


my_file['clean_url'] 


# In[14]:


text_train, text_test, y_train, y_test = train_test_split(my_file['clean_url'], 
                                                    my_file['Class'], 
                                                    test_size=0.20,
                                             stratify = my_file['Class'])


# In[15]:


# Tokenize and transform to integer index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X_train) # longest text in train set

# Add pading to ensure all vectors have same dimensionality
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[16]:



# Define CNN architecture

embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# Fit model
history = model.fit(X_train, y_train,
                    epochs=5,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=64)
loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[17]:



plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

get_ipython().run_line_magic('matplotlib', 'inline')
plot_history(history)


# In[18]:


New_URL = pd.Series(['www.facebook.com'])


# In[19]:


# vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,4),max_features=5000)

features_test = tfidf.transform(New_URL)


# In[ ]:


New_URL


# In[ ]:


predictions = model.predict(features_test)


# In[ ]:


predictions


# In[ ]:


get_ipython().system('pip install Keras')


# In[ ]:





# In[ ]:




