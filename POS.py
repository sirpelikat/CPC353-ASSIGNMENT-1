import pandas as pd
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv("CPC 353 Dataset - Sheet1.csv")

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

plt.savefig("Top Adjectives Bar Graph.png", dpi=300)
plt.clf()

# ---------- 2. Top Nouns ----------
noun_words, noun_counts = zip(*noun_freq)
plt.barh(noun_words, noun_counts, color='orange')
plt.gca().invert_yaxis()
plt.title("Top 10 Nouns")
plt.xlabel("Frequency")
plt.ylabel("Noun")
plt.tight_layout()
plt.show()

plt.savefig("Top Nouns Bar Graph.png", dpi=300)
plt.clf()

# ---------- 5. Optional: WordCloud for adjectives + nouns ----------
word_freq = dict(adj_freq + noun_freq)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Most Frequent Adjectives and Nouns")
plt.show()

wordcloud.to_file("Wordcloud of top nouns and adjectives.png")