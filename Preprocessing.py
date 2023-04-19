import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('JKII_Covid.csv')
df.head()
df = df.dropna()
data = df


df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df


data=data.reset_index()['Close']



##splitting dataset into train and test split
##splitting dataset into train and test split
training_size=int(len(data)*0.7)
test_size=len(data)-training_size
train_data,test_data=data[0:training_size],data[training_size:len(data)]

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, X=0,1,2,3-----99   Y=100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2..t+99 and Y=t+100



