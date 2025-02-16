import pandas as pd 
import numpy as np 
import mlflow 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split

# Initialize DagsHub integration and set the experiment for MLFlow tracking
dagshub.init(repo_owner='sfgrahman', repo_name='mlops_complete_project', mlflow=True)
mlflow.set_experiment("Experiment 2")
mlflow.set_tracking_uri("https://dagshub.com/sfgrahman/mlops_complete_project.mlflow")

# Load the dataset from csv file and split into training and testing sets
data = pd.read_csv(r"C:\Users\sfg\Downloads\water_potability.csv")
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

# Define a function to fill missing values with the median value for each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df
# Preprocess training and testing data to fill missing values
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Split the data into features (X) and target (y) for training and testing
X_train = train_processed_data.drop(columns =["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns =["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Define multiple baseline models to compare performance
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

# Start a parent MLFlow run to track the overall experiment
with mlflow.start_run(run_name="Water Potability Models Experiment"):
    # Iteratee over each model in the directory
    for model_name, model in models.items():
        # Start a child run within the parent run for each individual model
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on the training data
            model.fit(X_train, y_train)
            
            # Save the trained model using pickle
            model_filename = f"{model_name.replace(' ','_')}.pkl"
            pickle.dump(model, open(model_filename,"wb"))
            
            # Make predictions one the test data
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test,y_pred)
            
             # Log metrics to MLFlow for tracking
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1score)
            
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
            mlflow.sklearn.log_model(model, model_name.replace(' ','_'))

            # Log the source code file for reference
            mlflow.log_artifact(__file__)
            
            # Set tags for the run to provide additional metadata
            mlflow.set_tag("author","sfgrahman")
print("All models have been trained and logged as child runs successfully")
            



