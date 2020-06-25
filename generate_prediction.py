import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle


def predict_temp(inputDF):

	X = inputDF[['Present_Tmax','Present_Tmin','Solar radiation','DEM','lat','lon','Slope']]
	y = inputDF['Act_Next_Tmin'].values.reshape(-1,1)

	regressor = pickle.load(open('C:\\Users\\Junyao\\Documents\\GitHub\\Tableau-Python-Temperature-Forecast\\tempMinPredict.sav', 'rb'))
	y_pred = regressor.predict(X)
	tminDf = pd.DataFrame({'Actual_Next_Tmin': y.flatten(), 'Predicted_Next_Tmin': y_pred.flatten()})

	X = inputDF[['Present_Tmax','Present_Tmin','Solar radiation','DEM','lat','lon','Slope']]
	y = inputDF['Act_Next_Tmax'].values.reshape(-1,1)

	regressor = pickle.load(open('C:\\Users\\Junyao\\Documents\\GitHub\\Tableau-Python-Temperature-Forecast\\tempMaxPredict.sav', 'rb'))
	y_pred = regressor.predict(X)
	tmaxDf = pd.DataFrame({'Actual_Next_Tmax': y.flatten(), 'Predicted_Next_Tmax': y_pred.flatten()})

	dataDF = inputDF['Date']


	result = pd.concat([dataDF,tminDf,tmaxDf],axis=1,sort=False)
	result['id'] = np.arange(len(result))

	return result

def get_output_schema():       
	return pd.DataFrame({
		'Actual_Next_Tmin' : prep_decimal(),
		'Predicted_Next_Tmin' : prep_decimal(),
		'Actual_Next_Tmax' : prep_decimal(),
		'Predicted_Next_Tmax' : prep_decimal(),
		'id' : prep_int(),
		'Date' : prep_date()
	})