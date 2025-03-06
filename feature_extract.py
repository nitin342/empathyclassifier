import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("empatheticdialogues/cleaned_train.csv", on_bad_lines='skip')
df = df.dropna(subset=['utterance'])

# determine how many 

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['utterance'])

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# Get corresponding labels (emotion categories)
y = df['context']  # 'context' contains the emotion labels


print("TF-IDF transformation complete! Shape:", X_train_tfidf.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y, test_size=0.2, random_state=42)

# check how many rows for each emotion
print(y_train.value_counts())


print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

clf = LogisticRegression()
clf.fit(X_train, y_train)

train_score = 1 - clf.score(X_train, y_train)
val_score = 1 - clf.score(X_val, y_val)

# what is the most detected emotion in the validation set
y_pred = clf.predict(X_val)
print(pd.Series(y_pred).value_counts())

print("Training error:", train_score)
print("Validation error:", val_score)