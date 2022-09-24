import sklearn as skl
import sklearn.metrics
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, mean_squared_error,accuracy_score
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import joblib  # Save to file

total_data = pd.read_csv(r"Student-Performance\Code\student-mat.csv", sep = ';')
# we need to clean the data and get everything into a numeric form
col_to_remove = ['address', 'reason','romantic','Walc','nursery','traveltime','Mjob','Fjob']
data_drop = total_data.drop(col_to_remove, axis=1)

#encoding the data
label_encoder = preprocessing.LabelEncoder()

#creating list of features than needs to be encoded
encoded_features = ['school','sex','famsize','Pstatus','guardian','schoolsup','famsup','paid','activities','higher','internet']
#data_drop['school'] = label_encoder.fit_transform(data_drop['school'])
# encode the list of features with numeric values
data_drop[encoded_features] = data_drop[encoded_features].apply(LabelEncoder().fit_transform)
#split the data into X and y
X = data_drop.iloc[:,0:25]
y = data_drop.iloc[:,24]
print(data_drop,y)
#split the data in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

#we need to normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#training linear regression model
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Coefficients:',model.coef_)
print("Mean Sq Err: %.2f" % mean_squared_error(y_test,y_pred))
print("accuracy: ", model.score(X_train,y_train))
print("accuracy: ", model.score(X_test,y_test))
print(y_test)
print(y_pred)

# save the model to working directory
joblib_file = "best_model.pkl"   
joblib.dump(model, joblib_file)