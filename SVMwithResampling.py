#SVM with resampling

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate 
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.utils import resample

#load dataset
dfBreast = pd.read_csv('data.csv')

#Resampling dataset
value1 =dfBreast[dfBreast["diagnosis"]=="M"]
value0  = dfBreast[dfBreast["diagnosis"] == "B"]

diagnosis_downsample = resample(value1,
             replace=True,
             n_samples=285,)
diagnosis_upsample = resample(value0,
             replace=False,
             n_samples=284,)



dataBreast_downsampled = pd.concat([diagnosis_downsample, diagnosis_upsample])
dataBreast_downsampled["diagnosis"].value_counts()
#target value of diagnosis from categorical to numerical
dataBreast_downsampled['diagnosis'].replace(['M', 'B'],
                        [0, 1], inplace=True)

#remove not used features
df = dataBreast_downsampled.drop([dataBreast_downsampled.columns[0],dataBreast_downsampled.columns[-1],'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','symmetry_worst','fractal_dimension_worst'],axis=1)

Y = []
target = df['diagnosis']

for val in target:
    Y.append(val)

features = ['radius_worst', 'texture_worst', 'concave points_worst']
X = df[features].values.tolist()

## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test) 

#evaluation metrics on test set
precision, recall, fscore, support = score(y_test, y_pred)
print("accuracy: ",accuracy_score(y_test,y_pred))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))

#cross validation results on trainig set
print(cross_validate(clf, x_train, y_train, cv=10, scoring='accuracy',return_train_score=True))
print(cross_validate(clf, x_train, y_train, cv=10, scoring='recall',return_train_score=True))
