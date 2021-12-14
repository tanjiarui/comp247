import pandas as pd, graphviz, joblib  # , pandas_profiling as profile
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

# load and EDA
data_terry = pd.read_csv('student-por.csv', sep=';')
print(data_terry.info())
print(data_terry.isnull().sum())
print(data_terry.describe())
# report = profile.ProfileReport(data_terry)
# report.to_file('EDA')

# preprocessing
data_terry.insert(len(data_terry.columns), 'Pass', 0)
data_terry.Pass = data_terry.G1 + data_terry.G2 + data_terry.G3 >= 35
data_terry.Pass.replace([True, False], [1, 0], inplace=True)
data_terry.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
features_terry, target_terry = data_terry.drop('Pass', axis=1), data_terry.Pass
print(target_terry.value_counts())
numerical_col = list(features_terry.select_dtypes(exclude='object').columns)
categorical_col = list(features_terry.select_dtypes(include='object').columns)
transformer_terry = ColumnTransformer([('encoder', preprocessing.OneHotEncoder(), categorical_col)], remainder='passthrough')
clf_terry = DecisionTreeClassifier(criterion='entropy', max_depth=5)
pipeline_terry = Pipeline([('transformer', transformer_terry), ('decision tree', clf_terry)])
x_train_terry, x_test_terry, y_train_terry, y_test_terry = train_test_split(features_terry, target_terry, test_size=.2, random_state=48)

# modeling
pipeline = pipeline_terry.fit(x_train_terry, y_train_terry)
score = cross_val_score(pipeline_terry, x_train_terry, y_train_terry, cv=10)
print('accuracy: ', end='')
print(score)
print('mean accuracy: %f' % score.mean())
graph = export_graphviz(pipeline.steps[1][1], out_file=None)
graph = graphviz.Source(graph)
graph.render('decision tree', format='png')
predict = pipeline.predict(x_test_terry)
print(classification_report(y_test_terry, predict))

# fine tuning
param_grid = {'decision tree__min_samples_split': range(10, 300, 20), 'decision tree__max_depth': range(1, 30, 2), 'decision tree__min_samples_leaf': range(1, 15)}
grid_search = RandomizedSearchCV(pipeline_terry, param_grid, scoring='accuracy', cv=5, n_iter=7, refit=True, verbose=3)
grid_search.fit(x_train_terry, y_train_terry)
print('best parameter set:')
print(grid_search.best_params_)
print('best score: %f' % grid_search.best_score_)
predict = grid_search.predict(x_test_terry)
print(classification_report(y_test_terry, predict))
joblib.dump(grid_search.best_estimator_.steps[1][1], 'best model')
joblib.dump(grid_search.best_estimator_, 'pipeline')