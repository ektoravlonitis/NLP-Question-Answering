# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:53:36 2023

@author: spika
"""

import pandas as pd
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
class TextSearchEngine:
    def __init__(self, csv_file):
        self.lemmatizer = WordNetLemmatizer()
        self.df = pd.read_csv(csv_file, names=['index','number','title','body_text'], sep=',', skiprows=1, encoding='latin-1')
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)
        self.model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
        self.index = None
        self.process_data()

    def preprocess_text(self, text):
        line = text.strip().lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(line)
        words = [word for word in tokens if word.isalpha()]
        tokens_without_sw = [word for word in words if not word in stopwords.words('english')]
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens_without_sw]
        return " ".join(lemmatized)

    def process_data(self):
        # Here the data processing happens. It includes preprocessing of the raw text, TF-IDF vectorization and SBERT encoding
        # Preprocess the raw text
        self.df['body_text_processed'] = self.df['body_text'].apply(self.preprocess_text)
        # Transform the processed text into TF-IDF vectors
        X = self.vectorizer.fit_transform(self.df['body_text_processed'])
        self.df2 = pd.DataFrame(X.T.toarray(), index=self.vectorizer.get_feature_names_out())
        # Encode the raw text using Sentence-BERT (SBERT)
        encoded_data = [self.model.encode([text]) for text in self.df.body_text.tolist()]
        encoded_data = [np.asarray(item).astype('float32') for item in encoded_data]
        encoded_data = np.vstack(encoded_data)
        # Add the encoded data to the FAISS index for later retrieval
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.df))).astype(np.int64))


    def fetch_doc_info(self, dataframe_idx, score):
        # Fetches the document info given a DataFrame index
        info = self.df.iloc[dataframe_idx]
        meta_dict = dict()
        meta_dict['number'] = info['number']
        meta_dict['title'] = info['title']
        meta_dict['body_text'] = info['body_text'][:500]
        meta_dict['score'] = score
        return meta_dict

    # def search(self, query, top_k):
    #     # Search function uses Sentence-BERT to convert the query to a vector and then retrieves the most similar documents
    #     query_vector = self.model.encode([query])# Query is encoded using SBERT
    #     top_k = self.index.search(query_vector, top_k)# Most similar documents are retrieved from the FAISS index
    #     top_k_ids = top_k[1].tolist()[0]
    #     top_k_ids = list(np.unique(top_k_ids))
    #     results =  [self.fetch_doc_info(idx, score) for idx, score in zip(top_k_ids, top_k[0].tolist()[0])]
    #     return results
    
    #1st findingf the tf-idf n of the doc and then we are using sBert on the results of the tf-idf
    def search(self, query, top_k, tfidf_k=100):
        # TF-IDF Query
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        tfidf_scores = np.dot(self.df2.T.values, query_vector)
        top_k_tfidf_ids = np.argsort(tfidf_scores)[-tfidf_k:]
    
        # SBERT Query
        query_vector = self.model.encode([query])
        sbert_scores, sbert_ids = self.index.search(query_vector, top_k)
        top_k_sbert_ids = sbert_ids[0].tolist()
    
        # Intersection of top TF-IDF and SBERT ids
        common_ids = list(set(top_k_tfidf_ids).intersection(set(top_k_sbert_ids)))
    
        # Fetching results
        results = [self.fetch_doc_info(idx, score) for idx, score in zip(common_ids, sbert_scores[0])]
        results.sort(key=lambda x: x['score'], reverse=True)  # Sorting by score
    
        return results[:top_k]  # Returning top_k results



