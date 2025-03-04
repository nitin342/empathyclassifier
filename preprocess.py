import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure stopwords and tokenizer are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the dataset (modify path if needed)
df = pd.read_csv("empatheticdialogues/train.csv", on_bad_lines='skip')

# Keep only relevant columns
df = df[['utterance', 'context', 'prompt']]
df['utterance'] = df['utterance'].str.replace('_comma_', ' ')
df['context'] = df['context'].str.replace('_comma_', ' ')
df['prompt'] = df['prompt'].str.replace('_comma_', ' ')


# Predefined stopwords list
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return " ".join(tokens)

# Apply preprocessing to each text column
df['utterance'] = df['utterance'].apply(preprocess_text)
df['context'] = df['context'].apply(preprocess_text)
df['prompt'] = df['prompt'].apply(preprocess_text)

# Save the cleaned dataset (optional)
df.to_csv("empatheticdialogues/cleaned_train.csv", index=False)

print("Preprocessing complete. Cleaned data saved!")
