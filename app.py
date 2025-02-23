import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

data = load_diabetes()

x = data.data
y = data.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=23)

n_estimators = 10
lr = 0.1

clf = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=lr)

clf.fit(xtrain,ytrain)

ypred = clf.predict(xtest)

accuracy = accuracy_score(ytest,ypred)

matrix = confusion_matrix(ytest,ypred)

plt.figure(figsize=(5,5))

sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',xticklabels=data.target_names,yticklabels=data.target_names)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')

plt.savefig('matrix.png')

mlflow.set_experiment('demo')

with mlflow.start_run():

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('lr',lr)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_artifact('matrix.png')

    mlflow.sklearn.log_model(clf,'grandient bosting clf')

    mlflow.set_tag('author','akshat')
    mlflow.set_tag('project','demo')

    mlflow.log_artifact(__file__)