import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import dagshub

# Initialize DagsHub integration and set the experiment for MLFlow tracking
dagshub.init(repo_owner='sfgrahman', repo_name='mlops_complete_project', mlflow=True)
mlflow.set_experiment("Experiment 4")
mlflow.set_tracking_uri("https://dagshub.com/sfgrahman/mlops_complete_project.mlflow")

# Load the dataset from csv file and split into training and testing sets
data = pd.read_csv(r"C:\Users\sfg\Downloads\water_potability.csv")
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

# Define a function to fill missing values with the mean value for each column
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].mean()
            df[column].fillna(median_value,inplace=True)
    return df
# Preprocess training and testing data to fill missing values
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Split the data into features (X) and target (y) for training and testing
X_train = train_processed_data.drop(columns =["Potability"], axis=1)
y_train = train_processed_data["Potability"]

# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 4, 5, 6, 10]
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random forest model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, random_state=42)

#start a nw MLFlow run to log the Random Forest tuning process
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    
    #fit the RandomizedSearchCV object on the training data to identify the best parameters
    random_search.fit(X_train, y_train)
    
    # Log the parameters and mean test scores for each combination tried
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])

    # Print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", random_search.best_params_)
    
    # Log the best parameters in MLFlow
    mlflow.log_params(random_search.best_params_)
    
    # Train the model using the best parameters identified by RandomizedSearchCV
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)
    
     # Save the trained model to a file for later use
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare the test data by separating features and target variable
    X_test = test_processed_data.drop(columns=["Potability"], axis=1)  # Features
    y_test = test_processed_data["Potability"]  # Target variable

    # Load the saved model from the file
    model = pickle.load(open('model.pkl', "rb"))

    # Make predictions on the test set using the loaded model
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics: accuracy, precision, recall, and F1-score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics into MLflow for tracking
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    # Log the training and testing data as inputs in MLflow
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    
    mlflow.log_input(train_df, "train")  # Log training data
    mlflow.log_input(test_df, "test")  # Log test data

    # Log the current script file as an artifact in MLflow
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
    
    # Log the trained model in MLflow with its signature
    mlflow.sklearn.log_model(random_search.best_estimator_, "Best Model", signature=sign)

    # Print the calculated performance metrics to the console for review
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    
