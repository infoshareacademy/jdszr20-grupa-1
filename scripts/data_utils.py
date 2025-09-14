import pandas as pd
from langdetect import detect
from sklearn.model_selection import train_test_split

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def combine_datasets_and_export():
    d1 = pd.read_csv('../data_for_modeling/ClaimsKG_for_modeling.csv')
    d2 = pd.read_csv('../data_for_modeling/English_fake_for_modeling.csv')
    # syntetyczne dane - nie bierzemy pod uwagę
    # d3 = pd.read_csv('../data_for_modeling/fake-news-detection-for_modeling.csv')
    d4 = pd.read_csv('../data_for_modeling/ISOT_for_modeling.csv')
    d5 = pd.read_csv('../data_for_modeling/LIAR_for_modeling.csv')
    d6 = pd.read_csv('../data_for_modeling/WELFake_for_modeling.csv')
    # rename kolumny title_text to text - wyrzucilismy d3 wiec niepotrzebne
    # d3 = d3.rename(columns={'title_text': 'text'})
    # połącz title i text w kolumnę text
    d4['text'] = d4['title'] + ' ' + d4['text']
    d4 = d4.drop(columns=['title'])
    d4.head()
    dataset_all = pd.concat([d1, d2, d4, d5, d6], ignore_index=True)
    dataset_all.drop_duplicates(inplace=True)
    # wyrzuc nie-anglojęzyczne
    dataset_all = dataset_all[dataset_all['text'].apply(is_english)]
    # exportuj do csv
    dataset_all.to_csv('../data_for_modeling/dataset_all.csv', index=False)
    return dataset_all

def load_datasets(min_words: int = 5, max_words: int = 300) -> pd.DataFrame:
    dataset_all = pd.read_csv('../data_for_modeling/dataset_all.csv')
    # filtracja po liczbie słów
    dataset_all['word_count'] = dataset_all['text'].apply(lambda x: len(str(x).split()))
    dataset_all = dataset_all[(dataset_all['word_count'] >= min_words) & (dataset_all['word_count'] <= max_words)]
    dataset_all = dataset_all.drop(columns=['word_count'])
    return dataset_all

def load_dataset_sample(n: float = 0.3) -> pd.DataFrame:
    dataset_all = load_datasets()
    _, dataset_sample = train_test_split(dataset_all, train_size=n, stratify=dataset_all['fake'], random_state=42)
    return dataset_sample
