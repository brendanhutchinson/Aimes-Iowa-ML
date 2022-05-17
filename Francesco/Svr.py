from manipulation import *

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# selecting the features

num_svr = ['YearBuilt', 'YrSinceRm', 'TotalSF', 'BsmtUnfSF' , 'WoodDeckSF', '2ndFlrSF', 'ScreenPorch', 'GrLivArea', 
           'BsmtFinSF1', 'MasVnrArea', 'LotArea', 'Pool','YearRemodAdd','KitchenQual', 'Utilities', 'FullBath',
           'FireplaceQu','BsmtFinTotSF', 'GarageArea','TotalBath','ExterQual','GarageCars','1stFlrSF','OverallQual']

cat_svr = ['Neighborhood', 'MSSubClass', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 
           'Exterior2nd', 'HouseStyle', 'GarageType']

# dummyfing cathegorical features and creatinx X and y for traintest

y = df['SalePrice']

numeric = num_svr
categorical = cat_svr

dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)

# scaling X and y and splitting in train and test

y = np.array(y).reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_feat = sc_X.fit_transform(X)
y_feat = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_feat, y_feat.reshape(-1), test_size=0.30, random_state=42)

# setting, fitting and running the SVR

svr = SVR()
svr.set_params(kernel='rbf')
paramDict = {'C':np.linspace(1,100,20), 'gamma':np.linspace(1e-4, 1e-1, 10), 'epsilon':np.linspace(0.1,0.6,3)}
grid = GridSearchCV(svr, paramDict, cv=3, return_train_score = True)
ans_svr  = grid.fit(X_train, y_train)

# getting the best parameters, the best score, and R^2 for train and test set

ans_svr.best_params_  # 'C': 53.578947368421055, 'epsilon': 0.1, 'gamma': 0.001
print("The best score for SVR is: %.5f" % ans_svr.best_score_) # The best score for SVR is: 0.88946

svr_best = ans_svr.best_estimator_
svr_best.fit(X_train, y_train)
print("The R squared score for the test set is: %.5f" % (svr_best.score(X_train, y_train))) # 0.96889
print("The R squared score for the test set is: %.5f" % (svr_best.score(X_test, y_test))) # 0.93323