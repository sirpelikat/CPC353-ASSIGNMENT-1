import pandas as pd

# Load the dataset
df = pd.read_csv("CPC 353 Dataset - Sheet1.csv")

import spacy
from collections import Counter

# Load English model
nlp = spacy.load("en_core_web_sm")

# Combine all reviews into one text
doc = nlp(" ".join(df['REVIEW'].astype(str)))

# Extract adjectives and nouns
adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

# Count most common
adj_freq = Counter(adjectives).most_common(10)
noun_freq = Counter(nouns).most_common(10)

print("Top Adjectives:", adj_freq)
print("Top Nouns:", noun_freq)
