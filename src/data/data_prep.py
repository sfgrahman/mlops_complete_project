import pandas as pd 
import numpy as np 
import os 

def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
        

def fill_missing_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].mean()
                #df[column].fillna(median_value, inplace = True)
                df.fillna({column:median_value}, inplace = True)
        return df
    except Exception as e:
        raise Exception(f"Error Filling missing values: {e}")

def save_data(df: pd.DataFrame, filepath:str)->None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}:{e}")

def main():
    try:
        raw_data_path = "./data/raw/"
        train_data=load_data(os.path.join(raw_data_path, "train.csv"))
        test_data=load_data(os.path.join(raw_data_path, "test.csv"))
        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path)
        save_data (train_processed_data,os.path.join(data_path, "train_processed.csv"))
        save_data(test_processed_data,os.path.join(data_path, "test_processed.csv"))
    except Exception as e:
        raise Exception(f"An error occurred {e}")
    
if __name__=="__main__":
    main()