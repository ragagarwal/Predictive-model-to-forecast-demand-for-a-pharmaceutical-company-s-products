import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

sales_daily = pd.read_csv("internProj\salesdaily.csv")
df2 = pd.read_csv("internProj\saleshourly.csv")
df3 = pd.read_csv("internProj\salesmonthly.csv")
df4 = pd.read_csv("internProj\salesweekly.csv")

sales_daily['datum'] = pd.to_datetime(sales_daily["datum"], infer_datetime_format= True)
indexedData = sales_daily.set_index(['datum'])

from datetime import datetime
print(indexedData.head())

y = sales_daily.M01AB

features= ['datum']

X = sales_daily[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)

model.fit(train_X , train_y) 


prediction = model.predict(val_X)
print(prediction)
print("First in-sample predictions:", model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, prediction)

print(val_mae)