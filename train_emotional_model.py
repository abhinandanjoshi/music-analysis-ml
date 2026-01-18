from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error




def train_emotion_regressor(X, y):
model = Ridge(alpha=1.0)
model.fit(X, y)
predictions = model.predict(X)
print('RMSE:', mean_squared_error(y, predictions, squared=False))
return model
