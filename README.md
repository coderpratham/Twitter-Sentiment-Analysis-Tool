# Twitter Sentiment Analysis Tool

This project is a simple and effective **Sentiment Analysis Tool** built with **Python**, **NLTK**, and **Scikit-learn**. It analyzes the sentiment of social media text data (like tweets) and classifies them as **positive**, **negative**, or **neutral**.

## 🔍 Overview

The tool processes raw text data by performing:
- Tokenization
- Stop-word removal
- Lemmatization

Then it vectorizes the text using **TF-IDF** and uses a **Naive Bayes classifier** to predict the sentiment.

## 📁 Dataset

The model is trained on a [Dataset](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv), which contains 1.6 million tweets annotated for sentiment.

- File used: `training.1600000.processed.noemoticon.csv`
- Only `text` and `target` columns are used.
- Sampled 50,000 tweets for faster training and testing.

## 🛠️ Features

- Preprocesses social media text (removes URLs, mentions, hashtags)
- Supports live user input for prediction
- Real-time sentiment prediction
- Simple command-line interface

## 📦 Dependencies

Install the required Python packages using pip: 
```bash
pip install pandas numpy nltk scikit-learn
