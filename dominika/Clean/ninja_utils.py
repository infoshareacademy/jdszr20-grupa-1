import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
pd.options.mode.chained_assignment = None
# ploty analizy slow, dlugosci tekstow itp.
def plot_plots_dataset(df):
    plot_fake_counts(df)
    plot_number_of_words(df)
    plot_boxen(df)

def plot_fake_counts(df):
    sns.countplot(data=df, x='fake', hue='fake')
    plt.title("Ilosc fake/real news")
    plt.xlabel("Fake news")
    plt.ylabel("Count")
    plt.show()


def plot_number_of_words(df):
    count_of_words = [len(x) for x in df['text'].str.split()]
    words_df = pd.DataFrame([count_of_words, df['fake']]).transpose()
    words_df.columns = ['count', 'fake']
    words_df
    plt.figure(figsize=(6,4))
    ax = sns.barplot(data=words_df, x="fake", y="count", estimator="mean", hue='fake')

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", label_type="edge")

    plt.xlabel("Fake")
    plt.ylabel("Średnia liczba słów")
    plt.title("Średnia liczba słów dla fake/prawda")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.violinplot(data=words_df, x="fake", y="count", inner="box", hue='fake')

    plt.xlabel("Fake")
    plt.ylabel("Liczba słów")
    plt.title("Rozkład liczby słów")
    plt.show()

def plot_boxen(df):
    df['chars'] = df['text'].apply(len)
    df['sentences'] = df['text'].apply(lambda corpus: nltk.sent_tokenize(corpus)).apply(len)
    df['words'] = df['text'].apply(lambda document: nltk.word_tokenize(document)).apply(len)

    value_vars = ['sentences', 'chars', 'words']
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    for idx, var in enumerate(value_vars):
        sns.boxenplot(ax=ax[idx], x='fake', y=var, data=df, hue='fake')
    plt.tight_layout()
    plt.show()


# funkcje do lemtyzacji, usuwania stopwords, wektoryzacji
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(message):
    message = str(message)
    tokens = message.split()
    processed_tokens = []
    for word in tokens:
        # zmiana na małe litery
        word_lower = word.lower()
        # pomijamy stopwords
        if word_lower in stop_words:
            continue
        # usuwamy znaki interpunkcyjne i nawiasy
        word_clean = re.sub(r'[^a-zA-Z0-9]', '', word_lower)
        if word_clean:  # jeśli coś zostało po oczyszczeniu
            # lematyzacja
            word_clean = lemmatizer.lemmatize(word_clean)
            processed_tokens.append(word_clean)
    return ' '.join(processed_tokens)

# funkcje do trenowania modeli i ewaluacji
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, precision_score

def train_model_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # policz acuracy, precision i fbeta
    fbeta_res = fbeta_score(y_test, y_pred, beta=0.5)
    accuracy_res = accuracy_score(y_test, y_pred)
    precision_res = precision_score(y_test, y_pred)
    res_key = type(model).__name__
    res_val = f"fbeta: {fbeta_res:.2f}, accuracy: {accuracy_res:.2f}, precision: {precision_res:.2f}"
    print(f"{res_key}, {res_val}")
    return (res_key, res_val)