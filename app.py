import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:\\Users\\Junyao\\Documents\\GitHub\\Tableau-Python-Temperature-Forecast\\dataset2.csv')

print("\nModeling Actual Next Day's Minimum Temperature")

antMinMean = dataset['Act_Next_Tmin']
print("Actual Next Tmin Mean: " + str(antMinMean.mean()))
print("Mean's 10%: " + str(antMinMean.mean() * 0.1) + "\n")

X = dataset[['Present_Tmax','Present_Tmin','Solar_radiation','DEM','lat','lon','Slope']]
y = dataset['Act_Next_Tmin'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual_Next_Tmin': y_test.flatten(), 'Predicted_Next_Tmin': y_pred.flatten()})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
tminPredictDF = df

print("\n\nModeling Actual Next Day's Maximum Temperature")


antMaxMean = dataset['Act_Next_Tmax']
print("Actual Next Tmin Mean: " + str(antMaxMean.mean()))
print("Mean's 10%: " + str(antMaxMean.mean() * 0.1) + "\n")


X = dataset[['Present_Tmax','Present_Tmin','Solar_radiation','DEM','lat','lon','Slope']]
y = dataset['Act_Next_Tmax'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual_Next_Tmax': y_test.flatten(), 'Predicted_Next_Tmax': y_pred.flatten()})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

tmaxPredictDF = df

result = pd.concat([tminPredictDF,tmaxPredictDF],axis=1,sort=False)

result.to_csv("prediction_result.csv")