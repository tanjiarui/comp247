import pandas as pd, numpy as np, joblib
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

data_terry_df2 = pd.read_csv('breast_cancer.csv')
# preprocessing
data_terry_df2.replace('?', np.nan, inplace=True)
data_terry_df2.bare.astype('float')
data_terry_df2.drop('ID', axis=1, inplace=True)
pca = PCA(1)
size_shape = pca.fit_transform(data_terry_df2[['size', 'shape']])
print('explained ratio: %f' % pca.explained_variance_ratio_[0])
data_terry_df2.insert(1, 'size_shape', size_shape)
data_terry_df2.drop(['size', 'shape'], 1, inplace=True)
X, Y = data_terry_df2.loc[:, 'thickness':'Mitoses'], data_terry_df2['class'].replace([2, 4], [0, 1]).astype('category')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=48)

pipe_svm_terry = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='median')), ('scaler', StandardScaler()), ('svc', SVC(random_state=48))])
param_grid = {'svc__kernel': ['linear', 'rbf', 'poly'], 'svc__C': [.01, 0.1, 1, 10, 100], 'svc__gamma': [.01, .03, .1, .3, 1, 3], 'svc__degree': [2, 3]}
grid_search_terry = GridSearchCV(pipe_svm_terry, param_grid, scoring='accuracy', refit=True, verbose=3)
grid_search_terry.fit(x_train, y_train)
print('best parameter set:')
print(grid_search_terry.best_params_)
print('best estimator:')
print(grid_search_terry.best_estimator_)
predict = grid_search_terry.predict(x_test)
print('metric on test set\n', classification_report(y_test, predict))
best_model_terry = grid_search_terry.best_estimator_
joblib.dump(best_model_terry.steps[2][1], 'best_model')
joblib.dump(best_model_terry, 'best_pipeline')
