import pandas as pd  
import numpy as np 

def next_day_actual(inputDF):

	newDF = inputDF

	newDF['Act_Next_Tmax'] = inputDF['Present_Tmax'].shift(-1)
	newDF['Act_Next_Tmin'] = inputDF['Present_Tmin'].shift(-1)

	return newDF

def get_output_schema():       
	return pd.DataFrame({
		'Present_Tmax' : prep_decimal(),
		'Present_Tmin' : prep_decimal(),
		'DEM' : prep_decimal(),
		'lat' : prep_decimal(),
		'lon' : prep_decimal(),
		'Slope' : prep_decimal(),
		'Solar radiation' : prep_decimal(),
		'Act_Next_Tmax' : prep_decimal(),
		'Act_Next_Tmin' : prep_decimal(),
		'Date' : prep_date()
	})