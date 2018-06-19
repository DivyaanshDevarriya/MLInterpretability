# MLInterpretability
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

#Function to clean and preprocess both Train and Test files
def File_Cleaning_and_preprocessing(path):
    file=pd.read_csv(path)
    columns=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    
    #handling missing or nan values for Age,Fare,Embarked and Cabin
    file.Age=file.Age.fillna(value=file.Age.mean())
    file.Fare=file.Fare.fillna(value=file.Fare.mean())
    file.Embarked=file.Embarked.fillna(value=file.Embarked.mode()[0])
    file.Cabin = file.apply(lambda obs: "No" if pd.isnull(obs['Cabin']) else "Yes", axis=1)

    #preprocessing the textual columns/features
    le=preprocessing.LabelEncoder()
    Text_columns=['Name','Sex','Ticket','Cabin','Embarked']
    for i in Text_columns:
        file[i]=le.fit_transform(file[i])

    #Normalizing the features except PassengerId and Survived
    for i in columns[2:]:
                mean, std = file[i].mean(), file[i].std()
                file.loc[:, i] = (file[i] - mean)/std
    return file
    

TrainFile_for_BlackBoxModel=File_Cleaning_and_preprocessing('C:/Users/Home/Downloads/train.csv')
TestFile_for_Models=File_Cleaning_and_preprocessing('C:/Users/Home/Downloads/test.csv')

Train_features=TrainFile_for_BlackBoxModel.values[0:,2:] #Columns Passenger_Id (index 0) and Survived (index 1) are not included in the training features
Train_target=TrainFile_for_BlackBoxModel.values[:,1] #Survived (index 1) is the target feature

Test_features=TestFile_for_Models.values[0:,1:] 

clf_BlackBoxModel = svm.SVC()
clf_BlackBoxModel.fit(Train_features,Train_target)
prediction_of_BlackBoxModel=clf_BlackBoxModel.predict(Test_features)


clf_DecisionTree=tree.DecisionTreeClassifier()
clf_DecisionTree.fit(Train_features,Train_target)
prediction_of_DecisionTree=clf_DecisionTree.predict(Test_features)


clf_LinearModel=linear_model.LinearRegression()
clf_LinearModel.fit(Train_features,Train_target)
prediction_of_LinearModel=clf_LinearModel.predict(Test_features)




print("Surrogate Model: Linear Regression")
print("R2 score:",r2_score(prediction_of_BlackBoxModel,prediction_of_LinearModel))
rmse=sqrt(mean_squared_error(prediction_of_BlackBoxModel,prediction_of_LinearModel))
print("Root mean square error:",rmse)

print("-----------------------")

print("Surrogate Model: Decision Tree")
print("R2 score:",r2_score(prediction_of_BlackBoxModel,prediction_of_DecisionTree))
rmse=sqrt(mean_squared_error(prediction_of_BlackBoxModel,prediction_of_DecisionTree))
print("Root mean square error:",rmse)




