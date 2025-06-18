"""
MSc NLP Assignment: Exploring and Applying NLP Techniques
Author: Ashtosh Tiwari
"""

# === SECTION 1: Imports and Setup ===
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import pipeline
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Downloads
nltk.download('punkt')
nltk.download('wordnet')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# === SECTION 2: Core Text Processing and Tool Comparison ===
texts = [
    "I'm loving the new iPhone! The camera is amazing.",
    "Terrible customer service at the bank today.",
    "Can't wait for the weekend to start.",
    "The movie was too long and boring.",
    "Excellent food and great service at the restaurant!"
]

# --- NLTK Processing ---
nltk_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

nltk_results = []
for text in texts:
    tokens = word_tokenize(text)
    stemmed = [nltk_stemmer.stem(w) for w in tokens]
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    nltk_results.append({
        "tokens": tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    })

# --- SpaCy Processing ---
spacy_results = []
for text in texts:
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmatized = [token.lemma_ for token in doc]
    spacy_results.append({
        "tokens": tokens,
        "lemmatized": lemmatized
    })

# --- Visualization ---
def plot_token_freq(token_lists, title):
    all_tokens = [token.lower() for sublist in token_lists for token in sublist if token.isalpha()]
    freq = Counter(all_tokens)
    df = pd.DataFrame(freq.items(), columns=['Token', 'Count']).sort_values(by='Count', ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    plt.bar(df['Token'], df['Count'], color='skyblue')
    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_token_freq([r["tokens"] for r in nltk_results], "NLTK Token Frequency")
plot_token_freq([r["tokens"] for r in spacy_results], "SpaCy Token Frequency")


# === SECTION 3: Named Entity Recognition and Sentiment Analysis ===

# Load NER pipeline with a BERT-based model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

sentences = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Barack Obama was the 44th President of the United States.",
    "Google is based in Mountain View, California.",
    "Elon Musk founded SpaceX and co-founded Tesla.",
    "India won the Cricket World Cup in 2011.",
    "Facebook was rebranded as Meta in 2021.",
    "The Eiffel Tower is in Paris, France.",
    "Amazon has warehouses across the USA.",
    "Microsoft acquired LinkedIn for $26 billion.",
    "Tesla is revolutionizing the electric vehicle industry."
]

# Perform NER
ner_data = []
for sentence in sentences:
    entities = ner_pipeline(sentence)
    for entity in entities:
        ner_data.append({
            "Sentence": sentence,
            "Entity": entity['word'],
            "Label": entity['entity_group'],
            "Score": round(entity['score'], 2)
        })

# Convert NER results to DataFrame
ner_df = pd.DataFrame(ner_data)
print("\n=== Named Entity Recognition Results ===")
print(ner_df)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample Real-World Reviews
sample_reviews = [
    "I love this product, it's amazing!",
    "Worst purchase ever. Completely disappointed.",
    "Not bad, but could be better.",
    "Absolutely fantastic! Will buy again.",
    "It's okay, nothing special."
]

# Perform Sentiment Analysis
sentiment_results = sentiment_pipeline(sample_reviews)

# Create DataFrame for visualization
sentiment_df = pd.DataFrame({
    "Review": sample_reviews,
    "Sentiment": [r['label'] for r in sentiment_results],
    "Confidence": [round(r['score'], 2) for r in sentiment_results]
})

print("\n=== Sentiment Analysis Results ===")
print(sentiment_df)


# === SECTION 4: Word Embeddings and Language Understanding ===

corpus_text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: 
once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 
'and what is the use of a book,' thought Alice 'without pictures or conversation?'

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), 
whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, 
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 
'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, 
but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, 
and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, 
or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
"""

# Tokenize corpus
sentences = sent_tokenize(corpus_text)
tokenized_sentences = [
    [word for word in word_tokenize(sentence.lower()) if word.isalpha()]
    for sentence in sentences
]

# Train Word2Vec
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Find similar words
word = 'rabbit'
similar_words = model.wv.most_similar(word, topn=5)

print(f"\n=== Top 5 words similar to '{word}' ===")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")


# === SECTION 5: Mini NLP Project: Text Classification ===

# Sample spam dataset
data = {
    "text": [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
        "Hey, are we still meeting for lunch today?",
        "Urgent: Your account has been compromised. Reset your password immediately.",
        "Hi mom, just checking in. How are you?",
        "Free entry in 2 a weekly competition to win FA Cup final tkts. Text to enter.",
        "Can you send me the notes from class?",
        "Win a brand new iPhone by entering this contest!",
        "Don't forget our meeting at 3 PM.",
        "Limited time offer! Exclusive deal just for you.",
        "Let’s catch up soon. Miss talking to you!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
}

df = pd.DataFrame(data)

# Preprocess
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# Evaluation
print("\n=== Naive Bayes Classification Report ===")
print(classification_report(y_test, nb_preds))
print("Accuracy:", accuracy_score(y_test, nb_preds))

# Confusion Matrix
cm = confusion_matrix(y_test, nb_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
