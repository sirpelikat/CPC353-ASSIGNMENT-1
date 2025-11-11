import pandas as pd
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

# Download resources
nltk.download('punkt')
nltk.download('punkt_tab')  # needed for new NLTK versions
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("CPC 353 Dataset - Sheet1.csv")

# Combine all reviews
text = " ".join(df['REVIEW'].astype(str))

# Tokenize and remove stopwords
tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
filtered = [w for w in tokens if w not in stopwords.words('english')]

# Generate bigrams and trigrams
bigrams = list(ngrams(filtered, 2))
trigrams = list(ngrams(filtered, 3))

# Count most common n-grams
bigram_freq = Counter(bigrams).most_common(10)
trigram_freq = Counter(trigrams).most_common(10)

print("Top 10 Bigrams:", bigram_freq)
print("Top 10 Trigrams:", trigram_freq)

# ---------- 3. Top Bigrams ----------
bigram_words = [' '.join(b) for b, _ in bigram_freq]
bigram_counts = [count for _, count in bigram_freq]
plt.barh(bigram_words, bigram_counts, color='green')
plt.gca().invert_yaxis()
plt.title("Top 10 Bigrams")
plt.xlabel("Frequency")
plt.ylabel("Bigram")
plt.tight_layout()
plt.show()

# ---------- 4. Top Trigrams ----------
trigram_words = [' '.join(t) for t, _ in trigram_freq]
trigram_counts = [count for _, count in trigram_freq]
plt.barh(trigram_words, trigram_counts, color='purple')
plt.gca().invert_yaxis()
plt.title("Top 10 Trigrams")
plt.xlabel("Frequency")
plt.ylabel("Trigram")
plt.tight_layout()
plt.show()