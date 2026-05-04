import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def create_features(df):
    df = df.copy()
    
    def tenure_group(tenure):
        if tenure <= 12: return '0-12'
        elif tenure <= 24: return '13-24'
        elif tenure <= 48: return '25-48'
        elif tenure <= 60: return '49-60'
        else: return '60+'
    
    df['TenureGroup'] = df['tenure'].apply(tenure_group)
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure']
    df['AvgMonthlySpend'] = df['AvgMonthlySpend'].fillna(0).replace([np.inf, -np.inf], 0)
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['HasMultipleServices'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['ContractMonths'] = df['Contract'].map(contract_map)
    df['Contract_Tenure_Interaction'] = df['ContractMonths'] * df['tenure']
    
    return df

def preprocess_data(df, target='Churn', drop_cols=['customerID']):
    df_proc = df.copy()
    
    # Encode target
    le = LabelEncoder()
    df_proc[target] = le.fit_transform(df_proc[target])
    
    # Encode categorical
    categorical_cols = df_proc.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in drop_cols + [target]]
    
    df_final = pd.get_dummies(df_proc.drop(columns=drop_cols), columns=categorical_cols, drop_first=True)
    
    # Split
    X = df_final.drop(columns=[target])
    y = df_final[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns.tolist()
