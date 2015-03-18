import numpy as np
import sklearn as sk
import pandas as pd
from time import time
import argparse 
import math
from pprint import pprint
from scipy import stats
### scikit imports 
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier, RandomizedLogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, KFold, train_test_split
from sklearn.grid_search import GridSearchCV


def load_data(file_name):
	df = np.load(file_name)
	Y = df[:,0]
	X = df[:, 1:]
	return X, Y

class ml_model():
	params = {}
	def __init__(self, name, method, params_):
		self.name = name
		self.method = method
		ml_model.params[name] = params_
		self.m = (self.name, self.method)

        @staticmethod	
	def get_params(name):
		return ml_model.params[name] if name in ml_model.params else None


## voting classifier
## TODO: weighted voting classifier and weights would the cv score of each classifiers best parameters
class VotingClassifier():
	def __init__(self):
		self.classifiers = [] 
        def add(self, est, score):
                self.classifiers.append((score, est))
	def predict(self,  X_test, k=3):
		ans = None
                self.classifiers.sort(reverse = True)
                ests = self.classifiers[:k]
		for score, est in ests:
		        y_pred = classifer.predict(X_test)
		        if ans is None:
				 ans = y_pred
			else:
				 ans = hstack((ans, y_pred))
		pred,  count = stats.mode(ans, axis=1)
		return pred

def transform(X_train, X_test):
	        nr = X_train.shape[0] 
		nc = X_train.shape[1]
		X = np.vstack((X_train, X_test))
		df = pd.DataFrame(X)
		
		
		for j in range(54, 57):
			df[j] = df[j].apply(lambda x : math.log10(1+x))
		X = df.as_matrix()
	        return X[:nr], X[nr:]
			    


## feature transformtation
min_max_scaler = ml_model('min-max-scaler', MinMaxScaler(), {})
std_scaler = ml_model('std-scaler', StandardScaler(), {})

## feature selection techniques
vr = ml_model('var-threshold', VarianceThreshold(), {})
select_k = ml_model('kbest', SelectKBest(f_classif), {'k' : (200, 400, 500, 700)})
rand_lr = ml_model('rand-lr', RandomizedLogisticRegression(), {'selection_threshold' : (0.05, 0.1, 0.2, 0.25)})

## classfication techniques
gb_tree = ml_model('gb-tree', GradientBoostingClassifier(), {'max_features' : ('sqrt', None), 'n_estimators' : (300, 500), 'subsample' :(0.5, 0.75, 1),  'max_depth' : (5, 10, 15, 30, 40), 'min_samples_leaf' : (5, 10, 15, 25, 40) })
rf_tree = ml_model('rf-tree', RandomForestClassifier(), {'max_depth' : (2, 5, 10, 20, 30, 40, 50), 'min_samples_leaf' : (10, 15, 25, 40, 50, 60, 75), 'n_estimators' :(100, 200), 'max_features' : ('sqrt', 'log2', None)})

l1_lr = ml_model('l1-lr', LogisticRegression(penalty='l1'), {'C' : 10.0 ** np.arange(-2, 2)}  )
sgd_lr = ml_model('sgd-lr', SGDClassifier(loss='log'), {'penalty' : ('l1', 'elasticnet') , 'C' : 10.0 ** np.arange(0, 2)}  )
svm_rbf = ml_model('svm-rbf', SVC(kernel = 'rbf'), {'C' : 10.0 ** np.arange(0, 2), 'gamma' : 10.0 ** np.arange(-2,0)})
gnb = ml_model('gaussian-nb', GaussianNB(), {})

## generate various simple non-nested pipelines 
## nested pipeline future !!
## mix all kinds of crazy shit 
def gen_pipeline_steps():
  for ft in [min_max_scaler, std_scaler]:                           ## feature transformation / normalization 
     for fs in [vr, select_k]:
       	for clf in [gb_tree, rf_tree]:     ## classifer 
	     name = ft.name + ':' + fs.name + ':' + clf.name
	     yield (name, [ft.m, fs.m, clf.m])
  for ft in [min_max_scaler, std_scaler]:                           ## feature transformation / normalization 
       	for clf in [gb_tree, rf_tree]:     ## classifer 
	     name = ft.name + ':' + clf.name
	     yield (name, [ft.m, clf.m])
  ''' 
  for ft in [min_max_scaler]:
	for fs in [vr]:
	    for clf in [svm_rbf]:
	         for rfe in [rfe_fs]:
	                name = ft.name + ':' + fs.name + ':' + clf.name + ':' + rfe.name 
	                yield (name, [ft.m, fs.m, clf.m, rfe.m])
  '''       

## setup hyper-parameters for the grid search
## parameters are defined along with MLModel
def get_hyper_params(pipeline):
  hyper_param_grid = {}
  for est_name, est in pipeline:
     if est_name in ml_model.params:
	 for param_name, vals in ml_model.get_params(est_name).iteritems():
                hyper_param_grid[est_name + '__' + param_name] = vals
  return hyper_param_grid
 

def do_grid_search(pipeline, parameters, cross_v, X, Y):
	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipeline.steps])
	print("parameters:")
	pprint(parameters)
	grid_search = GridSearchCV(pipeline, parameters, refit = True, cv = cross_v, verbose=1, n_jobs=-1)
	t0 = time()
        grid_search.fit(X, Y)
	print("done in %0.3fs" % (time() - t0))
	print()
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))
	
	return grid_search.best_estimator_, best_parameters, grid_search.best_score_

def do_kaggle_submission(Y_pred, name):
	   
	 print("Doing Kaggle Submission...")
         fout = open(name + '.sub.csv', 'w')
         fout.write("ID,Category")
         for i in range(Y_pred.shape[0]):
	      fout.write("\n" + str(i+1) + "," + str(int(Y_pred[i])))
         fout.close() 

if __name__ == '__main__':
       X, Y   = load_data('train.npy')
       X_test, Y_test = load_data('test_distribute.npy')
       print "# of features - ", X.shape[1], "# of examples - ", X.shape[0]
       cv = ShuffleSplit(n = X.shape[0], n_iter=3, test_size=0.20)
       vc  = VotingClassifier()
       for name, steps in gen_pipeline_steps():
	       pipeline   = Pipeline(steps)
	       parameters = get_hyper_params(steps)
	       print "Doing Grid Search for ", name 
	       best_est, best_parameters, best_score  = do_grid_search(pipeline, parameters, cv,  X, Y) ## StratifiedKfold with k=4 is performed
               vc.add(best_est, best_score)
               '''
	       est = Pipeline(steps)
	       print "setting the best params for est"
               
	       best_param = {}
	       for param_name in sorted(parameters.keys()):
		 best_param[param_name] = best_parameters[param_name]
		 print("\t%s: %r" % (param_name, best_parameters[param_name]))
               #est.set_params(**best_param)
	       try:
	          x_new = est.fit_transform(X, Y)
	          print "# of features-", x_new.shape[1]
	       except:
	          est.fit(X, Y)
	       '''
	       Y_pred = best_est.predict(X_test)
               do_kaggle_submission(Y_pred, str(best_score) + "-" + name)
       Y_pred = vc.predict(X_test, 3)
       do_kaggle_submission(Y_pred, "voting")

       '''
       skf = StratifiedKFold(Y, n_folds=4)
       scores = {}
       for name in pipelines:
         scores[name] = []  
       for train, test in skf:
         X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
         for name, pipeline in pipelines.iteritems():
           clf = pipeline 
	   print X_train[2:5, 0:2]
	   print X_train.shape
	   x_new_train = clf.fit_transform(X_train, Y_train)
	   print X_train[2:5, 0:2]
	   print x_new_train.shape
	   
	   Y_pred = clf.predict(X_test)
	   print name, "F1-score", f1_score(Y_test, Y_pred), "report=" 
           print classification_report(Y_test, Y_pred)
	   scores[name].append(f1_score(Y_test, Y_pred))
       
       for name in scores:
             df = pd.DataFrame(scores[name], columns=['f1'])       
	     print name
             print df.describe()
	     print '-----'
       for name, pipeline in pipelines.iteritems():
	       pipeline.fit(X, Y)
       
       x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.97, random_state=12345)
       for name in pipelines:
	    x_new = pipelines[name].fit_transform(x_train, y_train)
	    print 'name-', name,
	    print '# of features-', x_new.shape
	    y_pred = pipelines[name].predict(x_test) 
	    scores[name] = f1_score(y_test, y_pred)
       print scores 
       '''
       	
