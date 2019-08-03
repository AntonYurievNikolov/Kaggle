# Import xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.layers import Dense,BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers
import keras as ks
from scipy import stats
from scipy.stats import norm, skew
#
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
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#SCALE THE TARGET
train["SalePrice"] = np.log1p(train["SalePrice"])

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test),sort=True).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

#Handling Missing data
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
skewness = skewness[abs(skewness) > 0.75]
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

###Model taken from "Stacked Regressions : Top 4% on LeaderBoard" Kernel
xgbreg=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


xgbreg.fit(X, y)
xgb_train_pred = xgbreg.predict(X)
xgb_pred = np.expm1(xgbreg.predict(test))


def get_new_model ():
    n_cols =X.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape = (n_cols,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
#TESTING FEW LR
#Optimizing the model
lr_to_test = [0.001, 0.002, 0.003]
early_stopping_monitor = EarlyStopping(patience=2)

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
keras_model.fit(X, y,epochs=200,callbacks=[early_stopping_monitor])
keras_pred=np.expm1(keras_model.predict(test))
#ks.models.save_model(keras_model,'200 epochs 0.074250 RMSE.h5')

keras_train = keras_model.predict(X)
keras_model.summary()
print('\n Keras MSE: %f\n'%mean_squared_error(y_train, keras_train))
print('\n Keras RMSE: %f\n'%np.sqrt(mean_squared_error(y_train, keras_train)))
print('\n XGB MSE: %f\n'%mean_squared_error(y_train, xgb_train_pred))
print('\n XGB RMSE: %f\n'%np.sqrt(mean_squared_error(y_train, xgb_train_pred)))
#Keras MSE:     0.005513
#Keras RMSE:    0.074250
#XGB MSE:       0.006425
#XGB RMSE:      0.080158

#TO DO - ADD THE FINAL MODEL BASED ON THE OUTPUTS FOR ENSEMBLE
ensemble = np.zeros(shape=(1459,1))
for i in range(1459) :
    ensemble[i] = keras_pred[i]*0.5 + xgb_pred[i]*0.5
    
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)