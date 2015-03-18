import itertools                                       ## exhaustive search for parameter grid 
from sklearn.metrics import accuracy_score             ## accuracy score 
from sklearn.externals.joblib import Parallel, delayed ## for creating parallel jobs and creating 
from sklearn.base import clone                         ## deep copy the estimator 
from sklearn.utils import random
import numpy as np

''' cross-validation not done in parallel . Note: sklearn does this step also in parallel. 
    grid_score func is outside GridSearchCV as Parallel util wont allow class function . 
'''
def grid_score(estm, X, y, params, cv):
               cnt   = 0.00
               score = 0.0 
               for train, test in  cv:
                  est = clone(estm)
                  est.set_params(**params)
                  est.fit(X[train], y[train])
                  score += accuracy_score(y[test], est.predict(X[test]))
                  cnt += 1.0
               return score/cnt, params

class GridSearchCV():
     def __init__(self, est, parameters, cv, n_jobs = -1 , refit = True, verbose=1, pre_dispatch = '2*n_jobs'):
                self.est          = est
                self.param_grid   = parameters
                self.cv           = cv
                self.n_jobs       = n_jobs
                self.refit        = refit
                self.verbose      = verbose
                self.score_func   = None
                self.pre_dispatch = pre_dispatch
                if self.score_func is None:
                      self.score_func = accuracy_score

     ''' fit the X, y ''' 
     def fit(self, X, y):
              assert(X.shape[0] == y.shape[0])
              ### grid_score is the function that will be executed in parallel 
              n_candidate = 1
              for val in self.param_grid.values():
                   n_candidate *= len(val)
              print 'Fitting %i folds for each of %i candidates, totalling %i fits' % (len(self.cv), n_candidate, n_candidate)
              out = Parallel(n_jobs = self.n_jobs, pre_dispatch = self.pre_dispatch, verbose=self.verbose)(delayed(grid_score)(clone(self.est), X, y, params, self.cv) for params in self._gen_params())
              best_score, best_params = sorted(out, lambda a, b : -cmp(a[0], b[0]))[0] ## replace by max or min-heap for giving top-k grids
              self.est.set_params(**best_params)
              if self.refit:
                   self.est.fit(X, y)
              self.best_estimator_ = self.est
              self.best_score_ = best_score

     ''' generate the exhaustive grid '''       
     def _gen_params(self):
             keys   = self.param_grid.keys()
             values = self.param_grid.values()
             for value in itertools.product(*values):
                  yield dict(zip(keys, value))

             

''' My implementation of sklearn K-Fold . 
    K-fold cross validation of choosing the model parameters. Usually combined with GridSearch. 
'''
class KFold(object):

     def __init__(self, n, n_folds = 5, shuffle=False, random_state=None):
        if n < 0:
                raise ValueError('number of samples are %i . should be non-negative' % (n))
        elif n_folds <= 1:
                raise ValueError('number of folds are %i . should be >= 2' %(n_folds))
        elif n < n_folds:
                raise ValueError('number of samples (%i) is  less than number of folds (%i)' % (n, n_folds)) 
        self.n            = n
        self.n_folds      = n_folds
        self.shuffle      = shuffle
        self.random_state = random_state
        self.idxs              = np.arange(n)
        if shuffle :
            rng = check_random_state(self.random_state)
            rng.shuffle(idxs)
       

     def __iter__(self):
        n     = self.n
        k     = self.n_folds 
        folds = []
        s     = 0  ## start 
        extra = n%k
        e     = n/k + (1 if extra > 0 else 0) ## end
        if extra > 0:
            extra -= 1
        folds.append(range(s, e))
        for i in range(1,k):
                 s = e
                 e = s + n/k + (1 if extra > 0 else 0)
                 if extra > 0:
                    extra -= 1
                 folds.append(range(s, e))

        for i in range(k):
              train = []; test = [];
              for j in range(k):
                   if i == j: 
                       test.extend(folds[j])
                   else:
                       train.extend(folds[j])
              yield self.idxs[train], self.idxs[test]

     def __len__(self):
        return self.n_folds
     def __repr__(self):
         return "k-fold with %i samples and %i folds" % (self.n, self.n_folds)
     def __str__(self):
         return "k-fold"
        
''' testontrain is naive of choosing the model parameters where we choose best param_grid that gives best score on train data'''
class TestOnTrain(object):
    def __init__(self, n):            
        if n < 0:
               raise ValueError("number of samples (%i) should be >=1 " % (n))
        self.n = n
    def __iter__(self):
        n = self.n 
        yield range(n), range(n)
    def __len__(self):
        return self.n
    def __repr__(self):
        return "test-on-train"
    def __str__(self):
        return "test-on-train"
       
