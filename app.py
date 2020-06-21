import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:\\Users\\Junyao\\Documents\\GitHub\\Tableau-Python-Temperature-Forecast\\dataset2.csv')

print(dataset.describe())

dataset.plot(x='Act_Next_Tmax', y='Act_Next_Tmin', style='o')  
plt.title('Minimum Temperature vs Maximum Temperature')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
#plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Present_Tmax'])

plt.show()