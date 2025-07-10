import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download only what we need
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding='latin-1',
                 names=['target', 'ids', 'date', 'flag', 'user', 'text'])

df = df[['target', 'text']]
df['target'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})

# Optional: use a sample for quick testing
df = df.sample(n=50000, random_state=42)

# Initialize tools
tokenizer = RegexpTokenizer(r'\b\w+\b')  # tokenize words only, ignore punctuation
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)  # Remove URLs, mentions, hashtags
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Function to predict sentiment of custom tweet
def predict_sentiment(tweet):
    clean_tweet = preprocess_text(tweet)
    vectorized = vectorizer.transform([clean_tweet])
    prediction = model.predict(vectorized)
    return prediction[0]


# Try it with your own input
while True:
    user_input = input("Enter your tweet (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted sentiment: {sentiment}\n")
