import pandas as pd

# Load the dataset
df = pd.read_csv("CPC 353 Dataset - Sheet1.csv")

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

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
