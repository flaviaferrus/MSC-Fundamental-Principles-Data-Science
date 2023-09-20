#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:41:43 2022

@author: flaviaferrusmarimon
"""

# -*- coding: utf-8 -*-
"""
This script is build to preprocess the jobs dataset according to 
conclusions extracted from the data exploration notebook.
"""

# Import libraries
import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords 

#from itertools import chain

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Import train dataset
df_train = pd.read_csv('./train.csv', index_col = 'Id')

#df_train = df_train.head()

# Import test dataset
df_test = pd.read_csv('./test.csv', index_col = 'Id')

#df_test = df_test.head()


# Location split
def loc_split(df):
    df['location'] = df['location'].replace(np.NAN, 'nan')
    df['country'] = np.NAN
    df['region'] = np.NAN
    df['city'] = np.NAN
    
    for i in range(df.shape[0]):
        df['location'].iloc[i] = df['location'].iloc[i].replace(" ", "").lower()
        chunk = df['location'].iloc[i].split(',')
        for j in range(len(chunk)):
            if j == 0:
                df['country'].iloc[i] = chunk[0]
            elif j == 1:
                df['region'].iloc[i] = chunk[1]
            elif j == 2:
                df['city'].iloc[i] = chunk[2]
            
    df.drop('location', axis = 1, inplace = True)
    
    # Replace np.NAN as a new category
    for feat in ['country', 'region', 'city']:
        df[feat] = df[feat].replace(np.NAN, 'nan')
    
    return df


# Removing features
def feat_rm(df, feats):
    for feat in feats:
        df.drop(feat, axis = 1, inplace = True)
    
    return df
    

# Scaling features
def feat_scaled(df, feat):
    feat_scaled = df[feat].copy()
    
    feat_scaled = (feat_scaled - feat_scaled.mean()) / feat_scaled.std()
    
    df[feat] = feat_scaled
    
    return df


# Fill missing values
def fill_na(df, feats):
    for feat in feats:
        df[feat] = df[feat].replace(np.NAN, 'nan')
    return df


df_train = feat_scaled(loc_split(feat_rm(df_train.copy(), ['department', 'salary_range', 'job_id'])), 'required_doughnuts_comsumption')
df_test = feat_scaled(loc_split(feat_rm(df_test.copy(), ['department', 'salary_range', 'job_id'])), 'doughnuts_comsumption')

print('First step: DONE')

# Feature Encoding
# Create new feature for company_profile


# Bag of words------------------------------------------------
bow_feats = ['industry', 'function', 'benefits',
            'requirements', 'company_profile',
            'title', 'description']

# Word extraction
def word_extraction(sentence):    
    #ignore = ['a', "the", "is"]    
    #print('sentence ' , sentence)
    words = re.sub("[^\w]", " ",  sentence).split()    
    cleaned_text = [w.lower() for w in words if w not in stopwords.words('english')]    
    return cleaned_text

# Apply tokenization to all sentences
def tokenize(sentences):    
    words = []
    for sentence in sentences:        
        w = word_extraction(sentence)        
        words.extend(w)            
        words = sorted(list(set(words)))    
    return words
    
# Vocabulary
#def generate_vocab(df1, df2, bow_feats):
def generate_vocab(df1, bow_feats):
    
    #allsen = [df1[feat].tolist() for feat in bow_feats]
    #allsen.append([df2[feat].tolist() for feat in bow_feats])
    #flat_allsen = list(chain(*allsen))
    #flat_allsen = df1[bow_feats].sum(axis = 1, skipna = True).tolist()
    
    flat_allsen = df1[bow_feats[0]].str.cat(df1[bow_feats[1:]], sep=' ').tolist()
    
    #print(flat_allsen)
    return tokenize(flat_allsen)
    
# Bag of words implementation to all sentences
def generate_bow(df, vocab, feats):
    # List of list of bag vectors
    bag_vects = []
    for feat in feats:
        # List of bag vectors per feature
        feat_vects = []
        for sentence in df[feat].tolist():        
            words = word_extraction(sentence)        
            bag_vector = np.zeros(len(vocab))        
            for w in words:            
                for i,word in enumerate(vocab):                
                    if word == w:                     
                        bag_vector[i] += 1     
            feat_vects.append(bag_vector)
        bag_vects.append(np.asarray(feat_vects))
    bag_vects = np.asarray(bag_vects)
    
    # Sum bag vectors to obtain the total word counter for each element
    s = bag_vects[0]
    for vect in bag_vects:
        s = s + vect
    s = s - bag_vects[0]
    
    # Split bag vectors
    split_bag = pd.DataFrame(s, columns = vocab)
    df = pd.concat([df, split_bag], axis = 1, join = 'inner')
    
    # Drop exciding features
    df = feat_rm(df.copy(), bow_feats)
    
    return df

df_train = fill_na(df_train.copy(), bow_feats)
df_test = fill_na(df_test.copy(), bow_feats)

#vocab = generate_vocab(df_train.copy(), df_test.copy(), bow_feats)
vocab = generate_vocab(df_train.copy(), bow_feats)

#print('Vocabulary: DONE')

#print(len(vocab))

df_train = generate_bow(df_train.copy(), vocab, bow_feats)
df_test = generate_bow(df_test.copy(), vocab, bow_feats)

print('BOW: DONE')


# One-Hot-----------------------------------------------------
onehot_feats = ['required_experience', 'required_education',
                'employment_type']

# One-Hot implementation
def generate_onehot(df, feats):
    return pd.concat([df, pd.get_dummies(df[feats], prefix = feats)], axis = 1, join = 'inner')

df_train = generate_onehot(df_train.copy(), onehot_feats)
df_test = generate_onehot(df_test.copy(), onehot_feats)

df_train = feat_rm(df_train.copy(), onehot_feats)
df_test = feat_rm(df_test.copy(), onehot_feats)

print('One-Hot: DONE')


# Hashing-----------------------------------------------------
hash_feats = ['country', 'region', 'city']

# Hash function
def StringHash(a, m=750, C=1024):
# m represents the estimated cardinality of the items set
# C represents a number that is larger than ord(c)
    hashes = []
    for w in a:
        hash = 0
        for i in range(len(w)):
            hash = (hash * C + ord(w[i])) % m
        hashes.append(hash)
    return pd.Series(hashes)

# Hash implementation
def generate_hash(df, feats):
    df[feats] = df[feats].apply(StringHash)
    return df

df_train = generate_hash(df_train.copy(), hash_feats)
df_test = generate_hash(df_test.copy(), hash_feats)

print('Hash: DONE')

#print(df_train)


#------------------------------------------------------------

# Save preprocessing
df_train.to_csv('clear_train.csv')
df_test.to_csv('clear_test.csv')

print('Save: DONE')


