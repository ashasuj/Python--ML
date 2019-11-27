import pandas as pd
from sklearn.tree import DecisionTreeClassifier #For Decision Tree Algorithm
from sklearn.model_selection import train_test_split #for splitting data set into training and testing data
from sklearn.metrics import accuracy_score #To measure accuracy of predictions

music_data = pd.read_csv('music.csv')
X = music_data.drop(['genre'], axis =1)        
y = music_data['genre']
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2) #using 20% training data, this returns a tuple
model = DecisionTreeClassifier()
model.fit(X_train,y_train) # Passing the training data
predictions = model.predict(X_test) #Passing the testing input data for making predictions
score = accuracy_score(y_test,predictions) #Compares expected scores with predictions and returns an accuracy score
score