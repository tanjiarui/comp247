import pandas as pd, numpy as np, matplotlib.pyplot as plt, pandas_profiling as profile
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# load and EDA
data_terry = pd.read_csv('breast_cancer.csv')
print(data_terry.info())
print(data_terry.isnull().sum())
print(data_terry.describe())
report = profile.ProfileReport(data_terry)
report.to_file('EDA')

# preprocessing
data_terry.bare.replace('?', np.nan, inplace=True)
data_terry.bare.astype('float')
data_terry.fillna(data_terry.bare.median(), inplace=True)  # only column bare has nan values
data_terry.drop('ID', axis=1, inplace=True)
pd.plotting.scatter_matrix(data_terry)
plt.show()
pca = PCA(1)
size_shape = pca.fit_transform(data_terry[['size', 'shape']])
print('explained ratio: %f' % pca.explained_variance_ratio_[0])
data_terry.insert(1, 'size_shape', size_shape)
data_terry.drop(['size', 'shape'], 1, inplace=True)
X, Y = data_terry.loc[:, 'thickness':'Mitoses'], data_terry['class'].replace([2, 4], [0, 1]).astype('category')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=48)

# modeling
for C, kernel in [(.1, 'linear'), (1, 'rbf'), (1, 'poly'), (1, 'sigmoid')]:
	svm_terry = SVC(C=C, kernel=kernel)
	svm_terry.fit(x_train, y_train)
	train_predict = svm_terry.predict(x_train)
	print('metric on training set for %s\n' % kernel, classification_report(y_train, train_predict))
	test_predict = svm_terry.predict(x_test)
	print('metric on test set for %s\n' % kernel, classification_report(y_test, test_predict))