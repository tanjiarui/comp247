import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# EDA
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
X, Y = mnist['data'], mnist['target']
print(X.loc[0, 'pixel1'].dtype, Y.dtype)
print(X.shape, Y.shape)
some_digit1 = X.loc[7] # 3
some_digit_image1 = some_digit1.values.reshape(28, 28)
some_digit2 = X.loc[5] # 2
some_digit_image2 = some_digit2.values.reshape(28, 28)
some_digit3 = X.loc[0] # 5
some_digit_image3 = some_digit3.values.reshape(28, 28)
plt.axis('off')
plt.imshow(some_digit_image1)
plt.show()
plt.axis('off')
plt.imshow(some_digit_image2)
plt.show()
plt.axis('off')
plt.imshow(some_digit_image3)
plt.show()

# preprocessing
Y = Y.astype(np.uint8)
Y = np.where(Y < 4, 0, Y)
Y = np.where((Y > 3) & (Y < 7), 1, Y)
Y = np.where(Y > 6, 9, Y)
plt.hist(Y)
plt.xlabel('class')
plt.ylabel('frequency')
plt.title('histogram of each class')
plt.show()
X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]

# naïve bayes
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, Y_train)
result = NB_classifier.predict([some_digit1, some_digit2, some_digit3])
print(result)
result = NB_classifier.predict(X_test)
print('accuracy on test set: '+str(accuracy_score(result, Y_test)))
accuracy = cross_val_score(NB_classifier, X_train, Y_train, cv=3, scoring='accuracy')
print('accuracy on cross validation: '+str(accuracy))

# logistic regression
LR_classifier = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=.1)
LR_classifier.fit(X_train, Y_train)
result = LR_classifier.predict(X_test)
print('accuracy of lbfgs solver on test set: '+str(accuracy_score(result, Y_test)))
print('classification metric:')
print(classification_report(Y_test, result))

LR_classifier_saga = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=.1, solver='saga')
LR_classifier_saga.fit(X_train, Y_train)
result = LR_classifier_saga.predict(X_test)
print('accuracy of saga solver on test set: '+str(accuracy_score(result, Y_test)))

result = LR_classifier.predict([some_digit1, some_digit2, some_digit3])
print(result)

accuracy = cross_val_score(LR_classifier, X_train, Y_train, cv=3, scoring='accuracy')
print('accuracy on cross validation: '+str(accuracy))