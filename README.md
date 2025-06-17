# Nlp
# MSc NLP Assignment - Natural Language Processing Pipeline

## Project Overview

This project is a comprehensive implementation of core NLP tasks as part of an MSc assignment, including:

1. **Core Text Processing and Tool Comparison**  
   - Tokenization, stemming, and lemmatization using NLTK and SpaCy  
   - Visualization of token frequencies  
   - Discussion of pros and cons of each tool

2. **Named Entity Recognition and Sentiment Analysis**  
   - Named Entity Recognition (NER) using a pre-trained BERT-based model from Hugging Face  
   - Sentiment analysis on real-world text samples using transformer models  
   - Visualization and interpretation of results

3. **Word Embeddings and Language Understanding**  
   - Training Word2Vec embeddings with Gensim  
   - Semantic similarity analysis  
   - Explanation of Skip-gram vs CBOW models

4. **Mini NLP Project: Text Classification**  
   - Preprocessing and cleaning a sample dataset  
   - Training a baseline Naive Bayes classifier  
   - Evaluation using accuracy, precision, recall, and confusion matrix  
   - Comparison with fine-tuned transformer models (optional)

---

## Project Structure

- `nlp_assignment.py` — Main Python script containing all tasks with detailed comments  
- Output CSV files for NER results, sentiment analysis, and classification dataset (generated during runtime)  
- Visualizations generated within the script (token frequency bar plots, confusion matrix, etc.)

---


#Install the python libraries
pip install nltk spacy gensim matplotlib seaborn transformers pandas scikit-learn
python -m spacy download en_core_web_sm
