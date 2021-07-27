import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv")
data = data[["class", "sms"]]

x = np.array(data["sms"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)

sample = input('Enter a message:')
data = cv.transform([sample]).toarray()

if clf.predict(data)=['spam']:
    print("The message is spam")
if clf.predict(data)=['ham']:
    print("The message is not spam")
