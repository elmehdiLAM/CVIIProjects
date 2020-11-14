from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# importing data
X, y = datasets.load_digits(return_X_y=True)
# spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

l = []
# looping in multiple case of K
for i in range(1, 10, 2):
    classifier = KNeighborsClassifier(i)
    score = cross_val_score(classifier, X_train, y_train, cv=5)
    l.append((np.mean(score), i))

# after the cross validation, choosing the best k
best_k = l[np.argmax([x[0] for x in l])][1]
print(f"the best choice of K: {best_k}")
# building the model with the best K
classifier = KNeighborsClassifier(best_k).fit(X_train, y_train)
# score of testing set
print(f"score of model on test data :{classifier.score(X_test, y_test)}")
# predicting test set show more details about error rate

from sklearn.metrics import classification_report
y_hat = classifier.predict(X_test)
print(classification_report(y_test, y_hat))

# predict from random number

random_int = np.random.randint(0, 1796)
print(random_int)
img = np.resize(X[random_int], (8, 8))
target = y[random_int]
pred = classifier.predict([X[random_int]])
print(f" real value of image is {target}")
print(f" predicted value of image is {pred}")

# predict you own number by passing it on data

