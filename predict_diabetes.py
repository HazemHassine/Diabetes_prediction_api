#Original file is located at
#    https://colab.research.google.com/drive/1wStx8xDTMeI6L1al2F0qDiMyR2P6Q6td

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np
import pickle as pkl

data_original = pd.read_csv("data.csv")

data_original.head(5)

data_original.describe().T

cols = data_original.columns[:-1]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data_cp = data_original.copy(deep=True)
X = pd.DataFrame(sc_X.fit_transform(data_cp.drop(["Outcome"],axis=1),),
                 columns=data_cp.columns[:-1])
# Get the y of X
y = data_cp["Outcome"]

# Check the scaled new valus
X.head(5)

# Splitting the data into train_x, train_y, test_x, test_y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=12, stratify=y)

#<u>**Note about Stratify parameter:**</u> stratifying the y means that we have the same proportions of each class (in out case it's **Outcome** and binary) in each the test and train data.

# checking the shape of our data
for i in [X_train,X_test,y_train,y_test]:
  print(i.shape)

# Choosing the model (knn in our case with the best k value)

from sklearn.neighbors import KNeighborsClassifier
# training then testing classifiers with different k values

train_sc = []
test_sc =[]
k_range = 30
for i in range(1, k_range):
  clf = KNeighborsClassifier(n_neighbors = i)
  clf.fit(X_train, y_train)
  train_sc.append(clf.score(X_train, y_train))
  test_sc.append(clf.score(X_test, y_test))

max_test_score = max(test_sc)
max_ind = test_sc.index(max_test_score)

print(f"The max score is 0.{int(max_test_score*100)} with a k value of {max_ind}")

#### The k we'll choose is 16 because that yields the best score

model = KNeighborsClassifier(n_neighbors = max_ind)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report")
print(classification_report(y_test,y_pred))

preds = model.predict([[0.1, 0.5, 0.9, .2,.1, 0.5, 0.9, .2,]])

print(preds)

with open("model_pkl.pkl", "wb") as f:
  pkl.dump(model, f)
