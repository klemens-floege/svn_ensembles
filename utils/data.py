import numpy as np
import pandas as pd
import torch
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sine_func(x):
    return np.sin(2 * np.pi * x)

def get_sine_data(n_samples, seed):
   
    rng = np.random.RandomState(seed)
    x_train = rng.uniform(0.0, 1.0, n_samples)
    y_train = sine_func(x_train) + rng.normal(scale=0.1, size=n_samples)
    x_test = np.linspace(0.0, 1.0, 100)
    y_test = sine_func(x_test)

    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, y_train, x_test, y_test

def get_gap_data(n_samples, seed, gap_start=-1, gap_end=1.5):
    """
    Generate synthetic data with a gap in the middle for uncertainty estimation.

    Parameters:
    num_samples_left (int): Number of samples to generate for the left side of the gap.
    num_samples_right (int): Number of samples to generate for the right side of the gap.
    gap_start (float): The starting point of the gap on the x-axis.
    gap_end (float): The ending point of the gap on the x-axis.

    Returns:
    numpy.ndarray: The x coordinates of the generated samples.
    numpy.ndarray: The y coordinates of the generated samples.
    """

    num_samples_left = int(n_samples*0.6)
    num_samples_right = n_samples - num_samples_left

    # Generate x coordinates on the left and right of the gap
    x_left = np.linspace(-3, gap_start, num_samples_left, endpoint=False)
    x_right = np.linspace(gap_end, 3, num_samples_right, endpoint=False)

    x_test = np.linspace(-3, 3, 100)
    
    # Combine both sets to form the full x dataset without the gap
    x_train = np.concatenate((x_left, x_right))
    
    # Generate the 'true' curve, which appears to be a combination of sinusoidal and polynomial
    y_true = np.sin(x_train) - np.power(x_train, 3) + np.cos(x_train-1)
    y_test = np.sin(x_test) - np.power(x_test, 3) + np.cos(x_test-1)
    
    # Add random noise to the 'true' curve to generate the observed y values
    noise = np.random.normal(0, 0.1, len(x_train))  # Noise level is lower for clearer visualization
    y_train = y_true + noise

    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return x_train, y_train, x_test, y_test


def load_yacht_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    data = pd.read_csv(file_path, sep=';')

    print(data.head())
    # Split the data into features and target variable
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable
    
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test


def load_energy_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    #data = pd.read_csv(file_path, sep=';')
    data = pd.read_excel(file_path)
    print(data.head())
    # Split the data into features and target variable
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable
    
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test


def load_autompg_data(test_size_split, seed, config):
   
    file_path =  config.task.file_path
    data = pd.read_excel(file_path)
    
    X = data.iloc[:, 1:-1].values  # Features: all columns except the first and last
    y = data.iloc[:, 0].values     # Target variable: the first column 'mpg'
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    
    return x_train, y_train, x_test, y_test


def load_concrete_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    data = pd.read_excel(file_path)
    print(data.head())
    
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test


def load_kin8nm_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    data = pd.read_excel(file_path)
    print(data.head())
    
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    
    return x_train, y_train, x_test, y_test


def load_naval_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    data = pd.read_excel(file_path)
    print(data.head())

    #TODO: double check this
    X = data.iloc[:, :-2].values  # Features
    y = data.iloc[:, -2].values   # Target variable

    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    
    return x_train, y_train, x_test, y_test

def load_power_data(test_size_split, seed, config):
    file_path =  config.task.file_path
    data = pd.read_excel(file_path)
    print(data.head())

    #PE is target variable
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    
    return x_train, y_train, x_test, y_test


def load_protein_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, 1:-1].values  # Features: all columns except the first and last
    y = data.iloc[:, 0].values     # Target variable: the first column 'RMSD'
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    
    return x_train, y_train, x_test, y_test

def load_wine_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    df = pd.read_csv(file_path, sep=';')
    
    # Preprocess the data
    X = df.drop(columns=['quality']).values
    y = df['quality'].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    return x_train, y_train, x_test, y_test

def load_parkinson_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_excel(file_path, skiprows=[0])
    #data = pd.read_excel(file_path)
    print(data.head())
    
    print("Dataset length:", len(data))
    print("Number of columns before one-hot encoding:", len(data.columns))

    unique_classes = data['status'].unique()
    n_classes = len(unique_classes)  # Count of unique classes
    print("Unique status classes:", unique_classes)
    print("Number of unique classes:", n_classes)

    #assert n_classes == config.task.dim_problem


    # One-hot encode the "status" column and update the dataframe
    encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    data_encoded = data.copy()  # Create a copy to keep the original data intact
    data_encoded['status'] = encoder.fit_transform(data[['status']]).ravel()  # Flatten array and assign

    # Separating features and target variable
    X = data_encoded.drop(columns=['status']).values  # Features: all columns except 'status'
    y = data_encoded['status'].values  # Target variable: encoded 'status'

    
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)

    
    return x_train, y_train, x_test, y_test


def load_breast_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_excel(file_path)
    #data = pd.read_excel(file_path)
    print(data.head())
    
    print("Dataset length:", len(data))
    print("Number of columns before one-hot encoding:", len(data.columns))

    unique_classes = data['Diagnosis'].unique()
    n_classes = len(unique_classes)  # Count of unique classes
    print("Unique status classes:", unique_classes)
    print("Number of unique classes:", n_classes)

    #assert n_classes == config.task.dim_problem


    # One-hot encode the "status" column and update the dataframe
    encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    data_encoded = data.copy()  # Create a copy to keep the original data intact
    data_encoded['Diagnosis'] = encoder.fit_transform(data[['Diagnosis']]).ravel()  # Flatten array and assign

    # Separating features and target variable
    X = data_encoded.drop(columns=['Diagnosis']).values  # Features: all columns except 'status'
    y = data_encoded['Diagnosis'].values  # Target variable: encoded 'status'

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)

    
    return x_train, y_train, x_test, y_test

def load_heart_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_excel(file_path)
    #data = pd.read_excel(file_path)
    print(data.head())
    
    print("Dataset length:", len(data))
    print("Number of columns before one-hot encoding:", len(data.columns))

    unique_classes = data['Heart Disease Presence'].unique()
    n_classes = len(unique_classes)  # Count of unique classes
    print("Unique status classes:", unique_classes)
    print("Number of unique classes:", n_classes)

    #assert n_classes == config.task.dim_problem


    # One-hot encode the "status" column and update the dataframe
    encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    data_encoded = data.copy()  # Create a copy to keep the original data intact
    data_encoded['Heart Disease Presence'] = encoder.fit_transform(data[['Heart Disease Presence']]).ravel()  # Flatten array and assign

    # Separating features and target variable
    X = data_encoded.drop(columns=['Heart Disease Presence']).values  # Features: all columns except 'status'
    y = data_encoded['Heart Disease Presence'].values  # Target variable: encoded 'status'

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)

    
    return x_train, y_train, x_test, y_test

def load_ionosphere_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_excel(file_path)
    #data = pd.read_excel(file_path)
    print(data.head())
    
    print("Dataset length:", len(data))
    print("Number of columns before one-hot encoding:", len(data.columns))

    unique_classes = data['Class'].unique()
    n_classes = len(unique_classes)  # Count of unique classes
    print("Unique status classes:", unique_classes)
    print("Number of unique classes:", n_classes)

    #assert n_classes == config.task.dim_problem


    # One-hot encode the "status" column and update the dataframe
    encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    data_encoded = data.copy()  # Create a copy to keep the original data intact
    data_encoded['Class'] = encoder.fit_transform(data[['Class']]).ravel()  # Flatten array and assign

    # Separating features and target variable
    X = data_encoded.drop(columns=['Class']).values  # Features: all columns except 'status'
    y = data_encoded['Class'].values  # Target variable: encoded 'status'

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)

    
    return x_train, y_train, x_test, y_test

def load_australian_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    data = pd.read_excel(file_path)
    #data = pd.read_excel(file_path)
    print(data.head())
    
    print("Dataset length:", len(data))
    print("Number of columns before one-hot encoding:", len(data.columns))

    unique_classes = data['Class'].unique()
    n_classes = len(unique_classes)  # Count of unique classes
    print("Unique status classes:", unique_classes)
    print("Number of unique classes:", n_classes)

    #assert n_classes == config.task.dim_problem


    # One-hot encode the "status" column and update the dataframe
    encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    data_encoded = data.copy()  # Create a copy to keep the original data intact
    data_encoded['Class'] = encoder.fit_transform(data[['Class']]).ravel()  # Flatten array and assign

    # Separating features and target variable
    X = data_encoded.drop(columns=['Class']).values  # Features: all columns except 'status'
    y = data_encoded['Class'].values  # Target variable: encoded 'status'

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)

    
    return x_train, y_train, x_test, y_test

def load_mnist_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    # Load data from Excel file
    if config.task.n_samples == True:
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path, nrows=config.task.n_samples)
    
    # Assuming the last column in the DataFrame is 'label'
    X = data.iloc[:, :-1].values  # Features (pixel values)
    y = data.iloc[:, -1].values   # Labels

    # Normalize pixel values to be between 0 and 1
    X = X / 255.0

    # Reshape the images from 784 pixels to 28x28 (if necessary, depending on model input requirement)
    # Adding a channel dimension for grayscale
    X_reshaped = X.reshape(-1, config.task.image_dim, config.task.image_dim)  # N, C, H, W format for PyTorch Conv2D

    # Convert labels and features into torch tensors
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # One-hot encode the labels
    #y_one_hot = torch.nn.functional.one_hot(y_tensor, num_classes=config.task.dim_problem)
    y_one_hot = y_tensor

    # One-hot encode the "status" column and update the dataframe
    #encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid multicollinearity by dropping first
    #data_encoded = data.copy()  # Create a copy to keep the original data intact
    #y = encoder.fit_transform(data_encoded.iloc[:, -1].values.reshape(-1, 1)).ravel()  # Flatten array and assign

    #y_one_hot = torch.tensor(y, dtype=torch.long)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_tensor, y_one_hot, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test

def load_fashionmnist_data(test_size_split, seed, config):
    file_path =  config.task.file_path 
    # Load data from Excel file
    if config.task.n_samples == True:
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path, nrows=config.task.n_samples)
    
    # Assuming the last column in the DataFrame is 'label'
    X = data.iloc[:, :-1].values  # Features (pixel values)
    y = data.iloc[:, -1].values   # Labels

    # Normalize pixel values to be between 0 and 1
    X = X / 255.0

    # Reshape the images from 784 pixels to 28x28 (if necessary, depending on model input requirement)
    X_reshaped = X.reshape(-1, config.task.image_dim, config.task.image_dim)  # N, C, H, W format for PyTorch Conv2D

    # Convert labels and features into torch tensors
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    y_one_hot = y_tensor

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_tensor, y_one_hot, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test

'''def load_cifar10_data(test_size_split, seed, config):
    file_path = config.task.file_path
    
    # Load data from Excel file based on the number of samples specified in the config
    if config.task.n_samples == 'all':
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path, nrows=config.task.n_samples)
    
    # Assuming the last column in the DataFrame is 'label'
    X = data.iloc[:, :-1].values  # Features (pixel values)
    y = data.iloc[:, -1].values   # Labels

    # Normalize pixel values to be between 0 and 1
    X = X / 255.0

    # Reshape the images from 3072 pixels to 32x32x3 for RGB images
    # CIFAR-10 images need to be reshaped to 3 channels (RGB)
    if config.task.image_dim == 32:
        X_reshaped = X.reshape(-1, 3, 32, 32)  # N, C, H, W format for PyTorch Conv2D
    else:
        raise ValueError("The CIFAR-10 image dimension is expected to be 32x32.")

    # Convert labels and features into torch tensors
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Optionally one-hot encode the labels, if needed
    if config.task.one_hot:
        y_tensor = torch.nn.functional.one_hot(y_tensor, num_classes=10)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=test_size_split, random_state=seed)
    
    return x_train, y_train, x_test, y_test'''



def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10_data(config):
    data_dir = config.task.file_path
    #load training data 
    for i in range(1, 6): 
        filename = data_dir+"data_batch_"+str(i) 
        dictionary = unpickle(filename) 
        x_data = dictionary[b'data']  
        y_data = np.array(dictionary[b"labels"])
        if i==1:   
            x_train = x_data  
            y_train= y_data  
        else:  
            x_train = np.concatenate((x_train, x_data), axis = 0)   
            y_train = np.concatenate((y_train, y_data), axis = 0)  
    #load testing data 
    filename = data_dir+"test_batch" 
    dictionary = unpickle(filename)
    data = dictionary[b"data"]
    x_test = data 
    y_test = np.array(dictionary[b"labels"]) 

    #normalise data: 
    x_train, x_test = x_train/ 255, x_test /255 
    
    if config.task.image_dim == 32:
        #x_train = x_train.reshape(-1, 3, 32, 32)  # N, C, H, W format for PyTorch Conv2D
        #x_test = x_test.reshape(-1, 3, 32, 32)  # N, C, H, W format for PyTorch Conv2D
        x_train = x_train.reshape(-1, 3, config.task.image_dim, config.task.image_dim) 
        x_test = x_test.reshape(-1, 3, config.task.image_dim, config.task.image_dim) 
        
    else:
        raise ValueError("The CIFAR-10 image dimension is expected to be 32x32.")

    #for debugging only
    #return x_train[:10], y_train[:10], x_test[:10], y_test[:10]

    return x_train, y_train, x_test, y_test
    