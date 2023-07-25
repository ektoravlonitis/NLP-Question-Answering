# Question Answering System for High School History Questions.

First run scraping.py, then IRmodel.py, and last generativeAi.py

Each csv file needed is created by the previous python file.

### Overview
In this project, we focus on developing an advanced question answering system tailored for high school students. The primary objective of the system is to efficiently answer history-related questions, providing accurate and relevant information to assist students in their learning journey. 

### Processing Steps
#### Web Scraping
In the web scraping phase, we programmatically extract relevant information from various web sources. This process involves crawling web pages, extracting HTML content, parsing the data, and transforming it into a structured format.

#### Information Retrieval (TF-IDF/SBERT)
The next step is information retrieval, where we use techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and SBERT (Sentence-BERT) to index and search the collected textual data effectively. TF-IDF calculates the importance of each word in a document corpus, while SBERT uses transformer-based models to generate context-aware embeddings for sentences or paragraphs. By using these methods, we can efficiently retrieve relevant documents or passages related to a given question.

#### Generative AI
The final step in the process involves generative AI, which aims to generate human-like responses to the given questions. The generative AI model learns from vast amounts of data and generates coherent and contextually relevant answers based on the input question. These models have the ability to understand language patterns, semantics, and context, allowing them to generate responses that appear natural and informative.