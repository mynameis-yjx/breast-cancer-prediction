import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
import seaborn as sns
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )
bc_df = pd.read_csv("wisc_bc_data.csv")
bc_df.head()
bc_df['diagnosis'].value_counts()
bc_df.shape
bc_df.dtypes
bc_df['diagnosis'] = bc_df.diagnosis.astype('category')
bc_df.dtypes
bc_df.head()
bc_df.describe().transpose()
bc_df.groupby(["diagnosis"]).count()
bc_df = bc_df.drop(labels = "id", axis = 1)
bc_df.shape
bc_feature_df = bc_df.drop(labels= "diagnosis" , axis = 1)
bc_feature_df.head()
bc_feature_df_z = bc_feature_df.apply(zscore)  # convert all attributes to Z scale

bc_feature_df_z.describe()

bc_labels = bc_df["diagnosis"]

X = np.array(bc_feature_df_z)
X.shape

y = np.array(bc_labels)
y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


NNH.fit(X_train, y_train)


predicted_labels = NNH.predict(X_test)
NNH.score(X_test, y_test)

from sklearn import metrics
print(metrics.confusion_matrix(y_test, predicted_labels))

bc_features_pruned_df_z =  bc_feature_df_z.drop(['radius_mean'], axis=1)
X = np.array(bc_features_pruned_df_z)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

NNH = KNeighborsClassifier(n_neighbors= 20 , weights = 'distance' )
NNH.fit(X_train, y_train)


predicted_labels = NNH.predict(X_test)


NNH.score(X_test, y_test)

from sklearn import metrics
print(pd.DataFrame(metrics.confusion_matrix(y_test, predicted_labels, labels=["M" ,"B"]), index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no']))

from sklearn.model_selection import cross_val_score

myList = list(range(1,50))

cv_scores = []
k_neighbors = []

for k in myList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    print(scores.mean())
    cv_scores.append(scores.mean())
    k_neighbors.append(k)
MSE = [1 - x for x in cv_scores]
print(min(MSE))
print(MSE.index(min(MSE)))
best_k = myList[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % best_k)
import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 18
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
plt.xlim(0,49)

plt.plot(k_neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
