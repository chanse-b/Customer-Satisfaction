import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
"""
referenced guide from https://towardsdatascience.com/data-preprocessing-with-python-pandas-part-1-missing-data-45e76b781993
"""

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

def getData():
    train_dataset_path = 'data/train.csv' 
    test_dataset_path = 'data/test.csv'  
    train_df = load_dataset(train_dataset_path)
    test_df = load_dataset(test_dataset_path)
    return train_df, test_df

def preprocess_data(df):
    print(df.head)
    # Drop any rows with missing data or NaN values
    df = df.dropna()

    # gender
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])        
    
    # type of customer
    df = pd.get_dummies(df, columns=['Customer Type'], drop_first=True)
    
    # age, scale to fit
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    
    # type of travel
    df = pd.get_dummies(df, columns=['Type of Travel'], drop_first=True)
    
    # class of travel
    df = pd.get_dummies(df, columns=['Class'])
    
    # flight distance
    df['Flight Distance'] = scaler.fit_transform(df[['Flight Distance']])
    
    # scaling departure delay and arrival delay
    df['Departure Delay in Minutes'] = scaler.fit_transform(df[['Departure Delay in Minutes']])
    df['Arrival Delay in Minutes'] = scaler.fit_transform(df[['Arrival Delay in Minutes']])
    
    # Map satisfaction to numerical values
    df['satisfaction'] = df['satisfaction'].map({
            'neutral or dissatisfied': 0,
            'satisfied': 1})
    
    #print(df.head)
    # Drop meaningless columns 
    df = df.drop(labels = ['Unnamed: 0','id'], axis = 1)

    # Separate features and target variable
    y = df['satisfaction']
    X = df.drop('satisfaction', axis=1)
    
    return X, y

#train, _ = getData()
#preprocess_data(train)
