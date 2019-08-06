# Import xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers
import keras as ks
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import lightgbm as lgb
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#We do not need the ID column
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

categorical_mask = (train.dtypes == object)
categorical_columns = train.columns[categorical_mask].tolist()
non_categorical_columns = train.columns[~categorical_mask].tolist()

###START OF FEATURE ENGINEERING  taken from "Stacked Regressions : Top 4% on LeaderBoard" Kernel
#Remove ouliers
#train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])

# will deal with more subtle outlers later.
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

# Removes outliers 
outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])
#SCALE THE TARGET
train["SalePrice"] = np.log1p(train["SalePrice"])

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test),sort=True).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

#Handling Missing data
# Not normaly distributed can not be normalised and has no central tendecy - Potentially will try this as well
#all_data = all_data.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#TRANSFORMING Features
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')


for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

#Fixing Skew  
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.50]#0.75 in the Original
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#ADDING Features 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]

####END OF FEATURE ENGINEERING
X = train
y =  y_train


#MODELS ###Models taken from "Stacked Regressions : Top 4% on LeaderBoard" Kernel
#Validation function
n_folds = 5
#HELPERS
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
 
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  
    
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))  

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
  
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

xgbreg=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#STACKING MODELS
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

#MY MODEL
def get_new_model ():
    n_cols =X.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape = (n_cols,)))
#    model.add(Dropout(rate = 0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
#TESTING FEW LR
#Optimizing the model
lr_to_test = [0.001, 0.002, 0.003]
early_stopping_monitor = EarlyStopping(monitor='mean_squared_error',patience=50)

#for lr in lr_to_test:
#    print('\n\nTesting model with learning rate: %f\n'%lr )
#    model = get_new_model()
#    my_optimizer = optimizers.Adam(lr=lr)
#    model.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics=['mse'])#For Categories and softmax metrics=['accuracy'])
#    model.fit(X, y,epochs=10, validation_split=0.3,callbacks=[early_stopping_monitor])
# 

from tensorflow import set_random_seed
set_random_seed(2)
#keras_model = ks.models.load_model('200 epochs 0.074250 RMSE.h5')
keras_model = get_new_model()  
my_optimizer = optimizers.Adam(lr=0.001)
keras_model.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics=['mse'])#For Categories and softmax metrics=['accuracy'])
keras_model.fit(X, y,
                epochs=1,#Add 200 Again when we use this again
                callbacks=[early_stopping_monitor])
keras_pred=np.expm1(keras_model.predict(test))
#ks.models.save_model(keras_model,'1000 epochs 0.060410 RMSE.h5')

#FIT AND PREDICT THE VALUES

stacked_averaged_models.fit(X.values, y)
stacked_train_pred = stacked_averaged_models.predict(X.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print('\n STACKED RMSLSE: %f\n'%rmsle(y, stacked_train_pred))


xgbreg.fit(X, y)
xgb_train_pred = xgbreg.predict(X)
xgb_pred = np.expm1(xgbreg.predict(test))
print('\n XGB RMSLSE: %f\n'%rmsle(y, xgb_train_pred))

model_lgb.fit(X, y)
lgb_train_pred = model_lgb.predict(X)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print('\n LGB RMSLSE: %f\n'%rmsle(y, lgb_train_pred))

keras_train = keras_model.predict(X)
print('\n Keras MSE: %f\n'%mean_squared_error(y_train, keras_train))
print('\n Keras RMSE: %f\n'%np.sqrt(mean_squared_error(y_train, keras_train)))

#The Ensemble score
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
#print(np.mean(np.expm1(y_train)-np.expm1((stacked_train_pred*0.70 +
#               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))))
#print(np.std(np.expm1(y_train)-np.expm1((stacked_train_pred*0.70 + 
#                   xgb_train_pred*0.15 + lgb_train_pred*0.15 ))))
#ADDING ADDITIONAL STACK FOR THE FINAL PRECITIONS
#ensembleDF = pd.DataFrame([stacked_train_pred,xgb_train_pred,lgb_train_pred,keras_train]).transpose()
#ensemblePredict = pd.DataFrame([stacked_pred,xgb_pred,lgb_pred,keras_pred]).transpose()
#better_ensemble = lasso#make_pipeline(RobustScaler(), LinearRegression())
#better_ensemble.fit(ensembleDF,y)
#better_ensemble_train_pred = better_ensemble.predict(ensembleDF)
#print('\n Other Ensemble RMSLSE: %f\n'%rmsle(y, better_ensemble_train_pred))
##0.072482
#ensemble = better_ensemble.predict(ensemblePredict)
#Better Ensemble?
#better_ensemble_predict = np.expm1(better_ensemble.predict(test.values))
#Keras MSE:     0.005513
#Keras RMSE:    0.074250
#XGB MSE:       0.006425
#XGB RMSE:      0.080158

#TO DO - ADD THE FINAL MODEL BASED ON THE OUTPUTS FOR ENSEMBLE
#ensemble = np.zeros(shape=(1459,1))
#for i in range(1459) :
#    ensemble[i] = keras_pred[i]*0.5 + xgb_pred[i]*0.5

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15 
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
#np.expm1(11.752738)-np.expm1(11.752738-0.115)
# np.expm1(11.752738)-np.expm1(11.752738-0.10626)
#sub['SalePrice'].mean()