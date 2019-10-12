import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table(filepath_or_buffer = 'F:/Spam-Classifier/smsspamcollection/SMSSpamCollection', 
	sep = '\t', names = ['label', 'sms_message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state = 1)

count_vectorizer = CountVectorizer()
training_data = count_vectorizer.fit_transform(X_train)
testing_data = count_vectorizer.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

print("Predictions : ", predictions)
print("Accuracy : ", accuracy_score(y_test, predictions))
print("Precision : ", precision_score(y_test, predictions))
print("Recall : ", recall_score(y_test, predictions))
print("F1 Score :", f1_score(y_test, predictions))
