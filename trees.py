#!/usr/bin/env python3

from manipulation import * 
from modeling import *


from sklearn import ensemble
randomForest = ensemble.RandomForestClassifier()
bagging = ensemble.BaggingClassifier()

randomForest.set_params(random_state=42, n_estimators=100, max_features=20)
randomForest.fit(X_train,y_train) 
randomForest.score(X_train,y_train)
randomForest.score(X_test,y_test)





# input which subset of numeric & categorical variables to be used 
numeric = num_cboost

categorical = cat_cboost


# DO NOT TOUCH THIS CODE 
dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)

# if subsetting use the code below before running tts
# X = X[subset]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


from manipulation import * 


from sklearn import ensemble
randomForest = ensemble.RandomForestRegressor()
bagging = ensemble.BaggingRegressor()

randomForest.set_params(random_state=42, n_estimators=100, max_features=20)
randomForest.fit(X_train,y_train) 
randomForest.score(X_train,y_train)
randomForest.score(X_test,y_test)

# bagging
bagging.set_params(n_estimators=50, random_state=42, max_features=20)
bagging.fit(X_train, y_train)
print("The training error is: %.5f" %(1 - bagging.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - bagging.score(X_test, y_test)))
print("The training accuracy is: %.5f" %( bagging.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(bagging.score(X_test, y_test)))

# Random Forest
randomForest.set_params(n_estimators=50, random_state=42, max_features=10,oob_score=True)
randomForest.fit(X_train, y_train)
print("The training error  is: %.5f" %(1 - randomForest.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - randomForest.score(X_test, y_test)))
print("The training accuracy is: %.5f" %(randomForest.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(randomForest.score(X_test, y_test)))



# Grid Search 
randomForest.set_params(oob_score=True)
grid_para_tree = {
    'max_depth': range(1, 30),
    'n_estimators': range(10, 110, 10),
    'min_samples_split': range(2,10,2)
    
}

randomForest.set_params(random_state=0)
grid_search_tree = GridSearchCV(randomForest.fit(X_train, y_train),grid_para_tree, scoring='r2', cv=5, n_jobs=-1, return_train_score=True)
%time grid_search_tree.fit(X_train, y_train)


print('Best parameters: '+ str(grid_search_tree.best_params_))
print(grid_search_tree.best_score_)
print(randomForest.oob_score_)




# Scoring test data set with tuned forest parameters 

randomForest.set_params(n_estimators=, random_state=42, max_features= ,oob_score=True)
randomForest.fit(X_train, y_train)
print("The training error  is: %.5f" %(1 - randomForest.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - randomForest.score(X_test, y_test)))
print("The training accuracy is: %.5f" %(randomForest.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(randomForest.score(X_test, y_test)))








# sorting feature importances 

tree_final = grid_search_tree.best_estimator_
feature_importance = list(zip(X_train.columns, tree_final.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
name, score = zip(*list(feature_sort))
pd.DataFrame({'feature':name,'score':score}).plot.barh(x='name', y='score')
















