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
# df = df[['utterance', 'context', 'prompt']]
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
    # Apply stemming
    tokens = [nltk.PorterStemmer().stem(word) for word in tokens]
    # Join tokens back into a string
    return " ".join(tokens)

# combine a whole conversation into one utterance
# extract from conv_id, format is hit:0_conv:1, where last digit is the conversation number

new_df = pd.DataFrame(columns=['conv_id', 'prompt', 'utterance', 'context'])

# iterate over the rows
for index, row in df.iterrows():
    conv_id = row['conv_id']
    conv_num = re.search(r'\d+$', conv_id).group()
    # check if the conversation number is the same as the previous row
    if index > 0 and conv_num == new_df.iloc[-1]['conv_id']:
        # combine the utterance with the previous row
        new_df.iloc[-1]['utterance'] += " " + row['utterance']
    else:
        new_row = {'conv_id': conv_num, 'prompt': row['prompt'], 'utterance': row['utterance'], 'context': row['context']}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)


# Apply preprocessing to each text column
new_df['utterance'] = new_df['utterance'].apply(preprocess_text)
# new_df['context'] = new_df['context'].apply(preprocess_text)
new_df['prompt'] = new_df['prompt'].apply(preprocess_text)

new_df.to_csv("empatheticdialogues/cleaned_train.csv", index=False)

print("Preprocessing complete. Cleaned data saved!")
