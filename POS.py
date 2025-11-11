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


# --- Visualization Section ---
import matplotlib.pyplot as plt

# Ensure plots display correctly
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 14


# ---------- 1. Top Adjectives ----------
adj_words, adj_counts = zip(*adj_freq)
plt.barh(adj_words, adj_counts)
plt.gca().invert_yaxis()
plt.title("Top 10 Adjectives")
plt.xlabel("Frequency")
plt.ylabel("Adjective")
plt.tight_layout()
plt.show()


# ---------- 2. Top Nouns ----------
noun_words, noun_counts = zip(*noun_freq)
plt.barh(noun_words, noun_counts, color='orange')
plt.gca().invert_yaxis()
plt.title("Top 10 Nouns")
plt.xlabel("Frequency")
plt.ylabel("Noun")
plt.tight_layout()
plt.show()
