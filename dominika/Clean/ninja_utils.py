import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
pd.options.mode.chained_assignment = None

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