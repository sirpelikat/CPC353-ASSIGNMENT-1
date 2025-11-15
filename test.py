import pandas as pd
import nltk
import spacy
import emoji
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

nltk.download('punkt')
nltk.download('punkt_tab')  # needed for new NLTK versions
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv("CPC 353 Dataset - Sheet1.csv")

# --------------------------------------------------------
# 1. REMOVE NON-ENGLISH REVIEWS
# --------------------------------------------------------
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

df['lang'] = df['REVIEW'].astype(str).apply(detect_language)
df = df[df['lang'] == 'en']   # keep only English reviews

# --------------------------------------------------------
# 2. REMOVE EMOJIS & SPECIAL CHARACTERS
# --------------------------------------------------------
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')  # remove emojis

df['clean_review'] = df['REVIEW'].astype(str).apply(remove_emojis)

# --------------------------------------------------------
# 3. LEMMATIZATION + TOKENS CLEANING
# --------------------------------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)

    cleaned_tokens = []

    for token in doc:
        # Conditions:
        # - alphabetical
        # - not stopword
        # - not punctuation
        # - keep lemma
        if token.is_alpha and token.lemma_ not in stop_words:
            cleaned_tokens.append(token.lemma_)

    return cleaned_tokens

df['tokens'] = df['clean_review'].apply(preprocess_text)

# --------------------------------------------------------
# 4. OPTIONAL: SENTENCE-LEVEL PROCESSING
#    (if needed for concordance or per-review analysis)
# --------------------------------------------------------
df['sentences'] = df['clean_review'].apply(lambda x: [sent.text for sent in nlp(x).sents])

# --------------------------------------------------------
# 5. REMOVE EXTREMELY SHORT REVIEWS
# --------------------------------------------------------
df = df[df['tokens'].apply(lambda x: len(x) > 2)]  # keep reviews with > 2 meaningful words

# --------------------------------------------------------
# OUTPUT CLEANED DATA
# --------------------------------------------------------
print("Remaining reviews after preprocessing:", len(df))
print(df.head())



