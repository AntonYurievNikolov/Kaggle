#Metrics
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

print('Sklearn MSE: {:.5f}. '.format(mean_squared_error(y_regression_true, y_regression_pred)))
print('Sklearn LogLoss: {:.5f}'.format(log_loss(y_classification_true, y_classification_pred)))


#EDA
# Calculate the ride distance
train['distance_km'] = haversine_distance(train)
plt.scatter(train.fare_amount, train.distance_km, alpha=0.5)
plt.xlabel('Fare amount')
plt.ylabel('Distance, km')
plt.title('Fare amount based on the distance')
plt.ylim(0, 50)
plt.show()

# Create hour feature
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['hour'] = train.pickup_datetime.dt.hour
# Find median fare_amount for each hour
hour_price = train.groupby('hour', as_index=False)['fare_amount'].median()

# Plot the line plot
plt.plot(hour_price.hour, hour_price.fare_amount, marker='o')
plt.xlabel('Hour of the day')
plt.ylabel('Fare amount')
plt.title('Fare amount based on day time')
plt.xticks(range(24))
plt.show()

#VALIDATION
# Import KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Create a StratifiedKFold object
str_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in str_kf.split(train, train.interest_level):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1
    
#HOW TO PREVENT TIME LEACKIGE
time_kfold = TimeSeriesSplit(n_splits=3)

# Sort train data by date
train = train.sort_values('date')

# Iterate through each split
fold = 0
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
    print('Fold :', fold)
    print('Train date range: from {} to {}'.format(cv_train.date.min(), cv_train.date.max()))
    print('Test date range: from {} to {}\n'.format(cv_test.date.min(), cv_test.date.max()))
    fold += 1
    
#FEATURE ENGINEERING
    # Concatenate train and test together
taxi = pd.concat([train, test])

# Convert pickup date to datetime object
taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])

# Create day of week feature
taxi['day_of_week'] = taxi['pickup_datetime'].dt.dayofweek

# Create hour feature
taxi['hour'] = taxi['pickup_datetime'].dt.hour

# Split back into train and test
new_train = taxi[taxi.id.isin(train.id)]
new_test = taxi[taxi.id.isin(test.id)]

#LABEL ENCODING

# Concatenate train and test together
houses = pd.concat([train, test])

# Label encode binary 'CentralAir' feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Create One-Hot encoded features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')

# Concatenate OHE features to houses
houses = pd.concat([houses, ohe], axis=1)

# Look at OHE features
print(houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))

#TARGET ENCODING
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

# Create mean target encoded feature
train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(train=train,
                                                                     test=test,
                                                                     target='SalePrice',
                                                                     categorical='RoofStyle',
                                                                     alpha=10)
# Look at the encoding
print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())


#FILLING MISSING VALUES
# Import SimpleImputer
from sklearn.impute import SimpleImputer

# Create mean imputer
mean_imputer = SimpleImputer(strategy='mean')

# Price imputation
rental_listings[['price']] = mean_imputer.fit_transform(rental_listings[['price']])
# Import SimpleImputer
from sklearn.impute import SimpleImputer
constant_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
rental_listings[['building_id']] = constant_imputer.fit_transform(rental_listings[['building_id']])

#PREDICTION
#NAIVE PREDICTIONS
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
naive_prediction = np.mean(validation_train['fare_amount'])
validation_test['pred'] = naive_prediction
rmse = sqrt(mean_squared_error(validation_test['fare_amount'], validation_test['pred']))
print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))

# Get pickup hour from the pickup_datetime column
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour
hour_groups = train.groupby('hour').fare_amount.mean()
test['fare_amount'] = test.hour.map(hour_groups)
test[['id','fare_amount']].to_csv('hour_mean_sub.csv', index=False)


from sklearn.ensemble import RandomForestRegressor
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'passenger_count', 'hour']
rf = RandomForestRegressor()
rf.fit(train[features], train.fare_amount)
test['fare_amount'] = rf.predict(test[features])
test[['id','fare_amount']].to_csv('rf_sub.csv', index=False)

#HYPERPARAMETER OPTIMIZATION
from itertools import product

# Hyperparameter grids
max_depth_grid = [3, 5, 7]
subsample_grid = [0.8, 0.9, 1.0]
results = {}
for max_depth_candidate, subsample_candidate in product(max_depth_grid, subsample_grid):
    params = {'max_depth': max_depth_candidate,
              'subsample': subsample_candidate}
    validation_score = get_cv_score(train, params)
    results[(max_depth_candidate, subsample_candidate)] = validation_score   
print(results)

#BLENDING
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
part_1, part_2 = train_test_split(train, test_size=0.5, random_state=123)
gb = GradientBoostingRegressor().fit(part_1[features], part_1.fare_amount)
rf = RandomForestRegressor().fit(part_1[features], part_1.fare_amount)
# Make predictions on the Part 2 data
part_2['gb_pred'] = gb.predict(part_2[features])
part_2['rf_pred'] = rf.predict(part_2[features])

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=False)
lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.fare_amount)
test['stacking'] = lr.predict(test[['gb_pred', 'rf_pred']])
print(lr.coef_)