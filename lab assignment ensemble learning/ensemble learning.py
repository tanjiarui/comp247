import pandas as pd  # , pandas_profiling as profile
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# load and EDA
columns = ['pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'bmi', 'diabetes pedigree function', 'age', 'class']
data_terry = pd.read_csv('pima-indians-diabetes.csv', names=columns)
print(data_terry.info())
print(data_terry.isnull().sum())
print(data_terry.describe())
# report = profile.ProfileReport(data_terry)
# report.to_file('EDA')

# preprocessing
features, target = data_terry.drop('class', axis=1), data_terry['class']
features = StandardScaler().fit_transform(features)
features = pd.DataFrame(features, columns=columns[:8])
x_train_terry, x_test_terry, y_train_terry, y_test_terry = train_test_split(features, target, test_size=.3, random_state=42)

# modeling
lr = LogisticRegression(max_iter=1400)
rf = RandomForestClassifier()
svm = SVC()
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=42)
extra_tree = ExtraTreesClassifier()
for classifier in [lr, rf, svm, decision_tree, extra_tree]:
	classifier.fit(x_train_terry, y_train_terry)
	predict = classifier.predict(x_test_terry[:3])
	print(classifier.__class__.__name__)
	print('predictions: %s, targets: %s' % (predict, list(y_test_terry[:3])))
# hard voting
hard_vote = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm), ('decision tree', decision_tree), ('extra tree', extra_tree)], voting='hard')
hard_vote.fit(x_train_terry, y_train_terry)
predict = hard_vote.predict(x_test_terry[:3])
print('hard voting classifier\npredictions: %s, targets: %s' % (predict, list(y_test_terry[:3])))
# soft voting
svm.probability=True
soft_vote = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm), ('decision tree', decision_tree), ('extra tree', extra_tree)], voting='soft')
soft_vote.fit(x_train_terry, y_train_terry)
predict = soft_vote.predict(x_test_terry[:3])
print('soft voting classifier\npredictions: %s, targets: %s' % (predict, list(y_test_terry[:3])))
# decision tree and extra tree
transformer_terry = ColumnTransformer([('scaler', StandardScaler(), columns[:8])], remainder='passthrough')
pipeline1_terry = Pipeline([('scaler', transformer_terry), ('extra tree', extra_tree)])
pipeline2_terry = Pipeline([('scaler', transformer_terry), ('decision tree', decision_tree)])
for pipeline in [pipeline1_terry, pipeline2_terry]:
	pipeline.fit(x_train_terry, y_train_terry)
	print(pipeline.steps[1][0])
	predict = pipeline.predict(x_test_terry)
	print(classification_report(y_test_terry, predict))
	score = cross_val_score(pipeline, x_train_terry, y_train_terry, cv=10)
	print('mean accuracy: %.2f' % score.mean())

# fine tuning
param_grid_48 = {'extra tree__n_estimators': range(10, 3000, 20), 'extra tree__max_depth': range(1, 1000, 2)}
grid_search_48 = RandomizedSearchCV(pipeline1_terry, param_grid_48)
grid_search_48.fit(x_train_terry, y_train_terry)
print('best parameter set:')
print(grid_search_48.best_params_)
print('best score: %f' % grid_search_48.best_score_)
predict = grid_search_48.predict(x_test_terry)
print(classification_report(y_test_terry, predict))