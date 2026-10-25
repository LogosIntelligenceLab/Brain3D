import numpy as np

def add_gaussian_noise(matrix, mean=0, std_dev=1):
    """Add Gaussian noise to the matrix"""
    gaussian_noise = np.random.normal(mean, std_dev, matrix.shape)
    return matrix + gaussian_noise

def add_rician_noise(matrix, sigma=1):
    """Add Rician noise to the matrix"""
    real_part = np.random.normal(0, sigma, matrix.shape)
    imaginary_part = np.random.normal(0, sigma, matrix.shape)
    rician_noise = np.sqrt((matrix + real_part)**2 + imaginary_part**2)
    return rician_noise

def add_drift_noise(matrix, drift_rate=0.01):
    """Add drift noise, representing slow variations over time or space"""
    drift_noise = np.linspace(0, drift_rate * matrix.size, matrix.size).reshape(matrix.shape)
    return matrix + drift_noise

def add_psychological_noise(matrix, fluctuation_intensity=0.5):
    """Add psychological noise, representing variability in signal response"""
    psychological_noise = np.random.uniform(-fluctuation_intensity, fluctuation_intensity, matrix.shape)
    return matrix + psychological_noise

# Example matrix representing fMRI signals
fMRI_matrix = np.random.rand(64, 64)  # 64x64 matrix for demonstration purposes

# Parameters for noise
gaussian_mean = 0
gaussian_std_dev = 0.1
rician_sigma = 0.1
drift_rate = 0.01
psychological_fluctuation_intensity = 0.2

# Add different types of noise
matrix_with_gaussian_noise = add_gaussian_noise(fMRI_matrix, gaussian_mean, gaussian_std_dev)
matrix_with_rician_noise = add_rician_noise(fMRI_matrix, rician_sigma)
matrix_with_drift_noise = add_drift_noise(fMRI_matrix, drift_rate)
matrix_with_psychological_noise = add_psychological_noise(fMRI_matrix, psychological_fluctuation_intensity)

# Output
print("Original Matrix:\n", fMRI_matrix)
print("\nMatrix with Gaussian Noise:\n", matrix_with_gaussian_noise)
print("\nMatrix with Rician Noise:\n", matrix_with_rician_noise)
print("\nMatrix with Drift Noise:\n", matrix_with_drift_noise)
print("\nMatrix with Psychological Noise:\n", matrix_with_psychological_noise)