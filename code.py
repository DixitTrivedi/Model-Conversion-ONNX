## IMPORT LIBRARIES
import pandas as pd
import pickle
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## LOAD DATA 
raw_df = pd.read_csv('data/Iris.csv')

## PREPROCESS ON DATA
processed_df = raw_df

[processed_df.rename(columns={col: col.lower()}, inplace=True) for col in processed_df.columns]

## SPLIT DATA
features = processed_df.drop(['species', 'id'], axis=1)
target = processed_df['species']
train_x, test_x, train_y,  test_y = train_test_split(features, target, test_size=0.2, random_state=42)

## SCALING DATA
sc = StandardScaler()
sc.fit(train_x)

train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

## ENCODING DATA
le = LabelEncoder()
le.fit(train_y)
train_y = le.transform(train_y)
test_y = le.transform(test_y)

## TRAIN MODEL
model = svm.SVC()
model.fit(train_x, train_y)

## PREDICTION
pred_y = model.predict(test_x)

## GET ACCURACY
acc = metrics.accuracy_score(pred_y, test_y)

## EXPORT MODEL
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(sc, open('model/scalar.pkl', 'wb'))

## PREDICT
model = pickle.load(open('model/model.pkl', 'rb'))
sc = pickle.load(open('model/scalar.pkl', 'rb'))

pred_data = sc.transform([[5.1, 3.5 ,1.4, 0.2]])

