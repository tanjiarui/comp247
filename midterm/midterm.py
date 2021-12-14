import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# load and EDA
df_terry = pd.read_csv('titanic_midterm_comp247.csv')
print(df_terry.info())
print(df_terry.isnull().sum())
print(df_terry.describe())
pd.plotting.scatter_matrix(df_terry)
plt.show()

# preprocessing
df_terry_features, df_terry_target = df_terry.loc[:, 'pclass':'embarked'], df_terry['survived']
numerical_cols = [col for col in df_terry_features.columns if df_terry_features[col].dtype in ['int','float']]
categorical_cols = [col for col in df_terry_features.columns if df_terry_features[col].dtype == 'object' and df_terry_features[col].nunique()<10]
x_train, x_test, y_train, y_test = train_test_split(df_terry_features, df_terry_target, test_size=.3, random_state=48)
num_pipeline_terry = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='median')), ('scaler', StandardScaler())])
cat_pipeline_terry = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')), ('encoder', OneHotEncoder())])
full_pipeline_terry = ColumnTransformer([('numerical', num_pipeline_terry, numerical_cols), ('categorical', cat_pipeline_terry, categorical_cols)])
x_train_transformed = full_pipeline_terry.fit_transform(x_train)

# modeling
clf_lr_terry = LogisticRegression(random_state=48)
clf_svm_terry = SVC(gamma='auto', random_state=48)
accuracy = cross_val_score(clf_lr_terry, x_train_transformed, y_train, cv=5)
print('accuracy of logistic regression on cross validation: %s' % accuracy)
print('mean accuracy of logistic regression on cross validation: %s' % accuracy.mean())
accuracy = cross_val_score(clf_svm_terry, x_train_transformed, y_train, cv=5)
print('accuracy of svm on cross validation: %s' % accuracy)
print('mean accuracy of svm on cross validation: %s' % accuracy.mean())

# fine tuning
param_grid = {'kernel': ['linear', 'rbf', 'poly'], 'C': [.01, 0.1, 1, 5], 'gamma': [.01, .02, .2, .3]}
grid_search = GridSearchCV(clf_svm_terry, param_grid, scoring='accuracy', refit=True, verbose=3)
grid_search.fit(x_train_transformed, y_train)
print('best parameters set:')
print(grid_search.best_params_)

# test
x_test_transformed = full_pipeline_terry.fit_transform(x_test)
predict = grid_search.predict(x_test_transformed)
print('metric on test set\n', classification_report(y_test, predict))
joblib.dump(grid_search.best_estimator_, 'best model')
joblib.dump(full_pipeline_terry, 'full pipeline')