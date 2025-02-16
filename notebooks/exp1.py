import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='sfgrahman', repo_name='mlops_complete_project', mlflow=True)

mlflow.set_experiment("Experiment1")

mlflow.set_tracking_uri("https://dagshub.com/sfgrahman/mlops_complete_project.mlflow")

data = pd.read_csv(r"C:\Users\sfg\Downloads\water_potability.csv")

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import  RandomForestClassifier
import pickle
X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 100

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    # save 
    pickle.dump(clf,open("model.pkl","wb"))

    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values
    #Import necessary metrics for evaluation
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    # Load the saved model for prediction
    model = pickle.load(open('model.pkl',"rb"))

    # Predict the target for the test data
    y_pred = model.predict(X_test)
    
    # calcualte performance metrics
    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    
    # Log metrics to MLFlow for tracking
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1score)
    
    # Log the number of estimators used as  a parameter
    mlflow.log_param("n_estimators", n_estimators)
    
    # Generate a confusion matrix to visualize model performance
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True) # visualize the confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matric")
    # Save the confusion matrix as PNG format
    plt.savefig("confusion_matrix.png")
    
    # Log the confusion matrix image to MLFlow
    mlflow.log_artifact("confusion_matrix.png")

    # Log the trained model to MLFlow
    mlflow.sklearn.log_model(clf,"RandomForestClassifier")

    # Log the source code file for reference
    mlflow.log_artifact(__file__)
    
    # Set tags in MLFlow to store additional metadata
    mlflow.set_tag("author","sfgrahman")
    mlflow.set_tag("model","RF")
    
    # Print out the performance metrics for reference
    print("acc",acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score",f1score)