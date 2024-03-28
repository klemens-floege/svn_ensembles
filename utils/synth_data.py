import numpy as np

def func(x):
    return np.sin(2 * np.pi * x)

def get_sine_data(n_samples, seed):
   
    rng = np.random.RandomState(seed)
    x_train = rng.uniform(0.0, 1.0, n_samples)
    y_train = func(x_train) + rng.normal(scale=0.1, size=n_samples)
    x_test = np.linspace(0.0, 1.0, 100)
    y_test = func(x_test)

    return x_train, y_train, x_test, y_test