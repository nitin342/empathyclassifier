import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def print_vectorizer_results(row):
    first_row = row.toarray()

    feature_names = count_vect.get_feature_names_out()

    tfidf_dict = {feature_names[i]: first_row[0][i] for i in np.nonzero(first_row)[1]}

    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 words in the first row with their TF-IDF scores:")
    for word, score in sorted_tfidf[:10]:
        print(f"{word}: {score:.4f}")


df = pd.read_csv("empatheticdialogues/cleaned_train.csv", on_bad_lines='skip')
df = df.dropna(subset=['utterance'])
X_train = df['utterance']
y_train = df['context']

valid_df = pd.read_csv("empatheticdialogues/cleaned_valid.csv", on_bad_lines='skip')
valid_df = valid_df.dropna(subset=['utterance'])

X_val = valid_df['utterance']
y_val = valid_df['context']

count_vect = CountVectorizer(ngram_range=(1, 3), stop_words='english', 
                             max_features=5000, min_df=2, analyzer='word')

X_train_counts = count_vect.fit_transform(X_train)
X_val_counts = count_vect.transform(X_val)

tfidf_transformer = TfidfTransformer(sublinear_tf=True)
X_train = tfidf_transformer.fit_transform(X_train_counts)
X_val = tfidf_transformer.transform(X_val_counts)

print_vectorizer_results(X_train_counts[0])

print("TF-IDF transformation complete! Shape:", X_train.shape)

# check how many rows for each emotion
# print(y_val.value_counts())


print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)

train_score = 1 - clf.score(X_train, y_train)
val_score = 1 - clf.score(X_val, y_val)

# what is the most detected emotion in the validation set
y_pred = clf.predict(X_val)
print(y_pred[-1])
# print(pd.Series(y_pred).value_counts())

print("Training error:", train_score)
print("Validation error:", val_score)