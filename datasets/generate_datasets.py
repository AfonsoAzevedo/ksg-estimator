import numpy as np
import os

def generate_bivariate_gaussian(n_samples, rho, random_state=42):
    """
    Generate samples from a bivariate Gaussian distribution.
    
    Why bivariate Gaussian?
    -----------------------
    This is the standard benchmark for MI estimators because the true MI
    has a known analytical formula:
        I(X;Y) = -0.5 * ln(1 - rho^2)
    where rho is the correlation coefficient.
    This lets us check not just whether Python and MATLAB agree,
    but whether both are actually correct.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    rho : float
        Correlation coefficient between X and Y.
        rho=0 means independent (true MI = 0)
        rho close to 1 means strongly dependent (true MI -> infinity)
    random_state : int
        Random seed so results are reproducible across runs.
    
    Returns
    -------
    x : np.ndarray of shape (n_samples, 1)
    y : np.ndarray of shape (n_samples, 1)
    true_mi : float
        Analytical ground truth MI in nats.
    """
    # Fix the random seed so we get the same data every run
    rng = np.random.default_rng(random_state)
    
    # Mean vector: both variables centered at zero
    mean = [0, 0]
    
    # Covariance matrix:
    # Diagonal entries are variances (both 1 here, so unit variance)
    # Off-diagonal entries are covariances (equal to rho since variances=1)
    # [[var(X),  cov(X,Y)],
    #  [cov(X,Y), var(Y) ]]
    cov = [[1,   rho],
           [rho, 1  ]]
    
    # Draw n_samples points from the 2D Gaussian
    # Result shape is (n_samples, 2)
    samples = rng.multivariate_normal(mean, cov, size=n_samples)
    
    # Split into separate column vectors
    # reshape(-1, 1) ensures shape is (n_samples, 1) not (n_samples,)
    # This matches MATLAB's column-wise convention
    x = samples[:, 0].reshape(-1, 1)
    y = samples[:, 1].reshape(-1, 1)
    
    # Compute the analytical ground truth MI
    # This is what both implementations should approximate
    true_mi = -0.5 * np.log(1 - rho**2)
    
    return x, y, true_mi


def generate_independent_uniform(n_samples, random_state=42):
    """
    Generate two completely independent uniform random variables.
    
    Why this test?
    --------------
    When X and Y are independent, true MI = 0 by definition.
    This is our simplest sanity check: if the estimator returns
    something far from zero here, something is wrong.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    x : np.ndarray of shape (n_samples, 1)
    y : np.ndarray of shape (n_samples, 1)
    true_mi : float
        Always 0.0 by definition.
    """
    rng = np.random.default_rng(random_state)
    
    # Draw x and y independently - no relationship between them
    x = rng.uniform(0, 1, size=(n_samples, 1))
    y = rng.uniform(0, 1, size=(n_samples, 1))
    
    # True MI is exactly zero since x and y are independent
    true_mi = 0.0
    
    return x, y, true_mi


def save_dataset(x, y, true_mi, filename):
    """
    Save a dataset to a CSV file so both Python and MATLAB
    can run on byte-for-byte identical data.
    
    File format:
    ------------
    First row: header (x, y)
    Remaining rows: one sample per row
    A separate .txt file stores the true MI value.
    
    Parameters
    ----------
    x : np.ndarray of shape (n_samples, 1)
    y : np.ndarray of shape (n_samples, 1)
    true_mi : float
    filename : str
        Path to save the CSV, e.g. 'datasets/data/gaussian_rho05.csv'
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Stack x and y side by side: shape (n_samples, 2)
    data = np.hstack([x, y])
    
    # Save as CSV with header
    # fmt='%.18e' saves full double precision so no information is lost
    np.savetxt(filename, data, delimiter=',', header='x,y',
               fmt='%.18e', comments='')
    
    # Save the true MI value in a separate file next to the CSV
    mi_filename = filename.replace('.csv', '_true_mi.txt')
    with open(mi_filename, 'w') as f:
        f.write(f'{true_mi:.18e}\n')
    
    print(f"Saved: {filename}")
    print(f"Saved: {mi_filename}")


if __name__ == "__main__":
    print("=== Generating and saving all datasets ===\n")
    
    # --- Independent uniform (simplest sanity check) ---
    print("--- Independent Uniform ---")
    x, y, true_mi = generate_independent_uniform(1000)
    save_dataset(x, y, true_mi, 'datasets/data/independent_uniform_n1000.csv')
    print(f"true MI = {true_mi:.4f}\n")
    
    # --- Bivariate Gaussian with varying correlation ---
    print("--- Bivariate Gaussian ---")
    for rho in [0.0, 0.5, 0.9]:
        x, y, true_mi = generate_bivariate_gaussian(1000, rho)
        # Build a clean filename from the rho value
        rho_str = str(rho).replace('.', '')
        filename = f'datasets/data/gaussian_rho{rho_str}_n1000.csv'
        save_dataset(x, y, true_mi, filename)
        print(f"rho={rho:.1f} | true MI={true_mi:.4f} nats\n")