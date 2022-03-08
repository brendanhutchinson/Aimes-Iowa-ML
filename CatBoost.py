
from trees import *

from trees import *
from catboost import CatBoostRegressor

# Cat Boost with defualt parameters 

model = CatBoostRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.score(X_test,y_test)


# Cat Boost Grid Search 

parameters = {
              'learning_rate': np.arange(.01,1),
              'max_depth': range(1,25),
               'n_estimators': [10,20,40,50,80,100,200,300]}
                 


grid_tree = GridSearchCV(model, param_grid= parameters, cv=3, n_jobs=-1).fit(X_train,y_train)


print('Best grid parameters: '+ str(grid_tree.best_params_))
print(grid_tree.best_score_)
print(randomForest.oob_score_)


# Scoring Test data set with tuned parameters 


model = CatBoostRegressor()
model.set_params(learning_rate = , max_depth= , n_estimators= )
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.score(X_test,y_test)

                