import numpy as np

def sine_func(x):
    return np.sin(2 * np.pi * x)

def get_sine_data(n_samples, seed):
   
    rng = np.random.RandomState(seed)
    x_train = rng.uniform(0.0, 1.0, n_samples)
    y_train = sine_func(x_train) + rng.normal(scale=0.1, size=n_samples)
    x_test = np.linspace(0.0, 1.0, 100)
    y_test = sine_func(x_test)

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
    
    return x_train, y_train, x_test, y_test