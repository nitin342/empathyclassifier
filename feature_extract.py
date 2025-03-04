import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("empatheticdialogues/cleaned_train.csv", on_bad_lines='skip')
df = df.dropna(subset=['utterance'])

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most important words

# Fit and transform the 'utterance' column into TF-IDF vectors
X = vectorizer.fit_transform(df['utterance'])

# Convert to dense format (optional, but useful for some models)
X = X.toarray()

# Get corresponding labels (emotion categories)
y = df['context']  # 'context' contains the emotion labels

print("TF-IDF transformation complete! Shape:", X.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)
val_score = clf.score(X_val, y_val)

print("Training accuracy:", train_score)
print("Validation accuracy:", val_score)