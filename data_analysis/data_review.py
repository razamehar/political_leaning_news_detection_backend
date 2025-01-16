import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from matplotlib import pyplot as plt
import string


def freq_dist(df, political_leaning_code):

    all_tokens = [token for sublist in df[df['political_leaning_encoded'] == political_leaning_code]['headline_tokens'] for token in sublist]
    freq_dist = FreqDist(all_tokens)

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    punctuation.update(['’', '‘', "'s", "n't", "''", "``"])

    filtered_tokens = [
        token.lower() for token in all_tokens
        if token.lower() not in stop_words and token not in punctuation
    ]

    freq_dist_updated = FreqDist(filtered_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(*zip(*freq_dist.most_common(10)), color='skyblue')
    axes[0].set_title('Original Token Frequency Distribution')
    axes[0].set_xlabel('Tokens')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticklabels([token for token, _ in freq_dist.most_common(10)], rotation=45, ha='right')

    axes[1].bar(*zip(*freq_dist_updated.most_common(10)), color='skyblue')
    axes[1].set_title('Filtered Token Frequency Distribution')
    axes[1].set_xlabel('Tokens')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticklabels([token for token, _ in freq_dist_updated.most_common(10)], rotation=45, ha='right')

    plt.tight_layout()
    return filtered_tokens


# Data load and review
df = pd.read_parquet("file.parquet")
print(df.sample(5))
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df['outlet'].unique())
print(df['outlet'].value_counts())
print(df['outlet'].value_counts().plot(kind='bar', color='skyblue'))
print(df['political_leaning'].value_counts())
print(df['political_leaning'].value_counts().plot(kind='bar', color='skyblue'))
print(df[df['political_leaning'] == 'UNDEFINED']['outlet'].value_counts())
print(df[df['political_leaning'] == 'UNDEFINED']['outlet'].value_counts().plot(kind='bar', color='skyblue'))

# Data Preprocessing
df = df.dropna()
print(df.shape)
df = df[df['political_leaning'] != 'UNDEFINED']
print(df.shape)

# Data Analysis & Visualization
print(df['political_leaning'].value_counts())
df[df['political_leaning'] == 'CENTER']['outlet'].value_counts().plot(kind='bar', color='skyblue')
df[df['political_leaning'] == 'RIGHT']['outlet'].value_counts().plot(kind='bar', color='skyblue')
df[df['political_leaning'] == 'LEFT']['outlet'].value_counts().plot(kind='bar', color='skyblue')

# Label Encoding
label_encoder = LabelEncoder()
df['political_leaning_encoded'] = label_encoder.fit_transform(df['political_leaning'])
print(df[['political_leaning', 'political_leaning_encoded']].sample(5))
df.drop(['outlet', 'lead','political_leaning'], axis=1, inplace=True)

# Feature Engineering
df['headline_count'] = df['headline'].apply(lambda x: len(x.split()))
df['body_count'] = df['body'].apply(lambda x: len(x.split()))

print(df['headline_count'].min())
print(df['headline_count'].max())
df['headline_count'].hist(color='skyblue')

print(df['body_count'].min())
print(df['body_count'].max())
df['body_count'].hist(color='skyblue')

# Word Frequency Analysis
#df['body_tokens'] = df['body'].apply(word_tokenize)
df['headline_tokens'] = df['headline'].apply(word_tokenize)
print(df.sample(5)

freq_dist(df, 0)
freq_dist(df, 1)
freq_dist(df, 2)

df.drop(['headline_count', 'body_count','headline_tokens'], axis=1, inplace=True)
print(df.sample(5))

# Manage Class Imbalance
min_class_size = df['political_leaning_encoded'].value_counts().min()
sampled_df = df.groupby('political_leaning_encoded', group_keys=False).apply(lambda x: x.sample(n=min_class_size))
balanced_sample = sampled_df.sample(frac=0.1, random_state=42)
print(balanced_sample['political_leaning_encoded'].value_counts())




