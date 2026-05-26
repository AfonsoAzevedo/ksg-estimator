import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma


def mi_knn(x, y, K=2):
    """
    Estimate Mutual Information I(X;Y) using the KSG estimator (Algorithm 1).

    This is a translation of Dr. AlMomani's MATLAB miKnn function.

    The KSG estimator works in four steps:
        1. Build the joint space (x,y) and find the k-th nearest neighbor
           distance for each point using the L-infinity norm.
        2. Use those distances as radii and count how many neighbors
           each point has in the marginal spaces x and y separately.
        3. Plug the neighbor counts into the KSG formula using
           the digamma function to get the MI estimate.

    The formula is:
        I(X;Y) = psi(K) + psi(N) - mean(psi(nx+1) + psi(ny+1))
    where psi is the digamma function.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, dx)
        First variable. Can be multivariate.
    y : np.ndarray of shape (n_samples, dy)
        Second variable. Can be multivariate.
    K : int
        Number of nearest neighbors to use.
        Higher K = more smoothing = lower variance but higher bias.
        K=2 is the default used in Dr. AlMomani's implementation.

    Returns
    -------
    I : float
        Estimated mutual information in nats.
    """

    # Input validation
    # Make sure x and y have the same number of samples
    assert x.shape[0] == y.shape[0], "x and y must have the same number of rows"

    # Ensure both are 2D arrays (n_samples, n_dims)
    # This matches MATLAB's column-wise matrix convention
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Total number of samples
    N = x.shape[0]

    # Step 1: Build joint space and find k-th neighbor distances
    # Concatenate x and y horizontally to form the joint space
    # If x is (N,1) and y is (N,1), JS is (N,2)
    # This matches MATLAB: JS = [x, y]
    JS = np.hstack([x, y])

    # Compute all pairwise distances in joint space using L-infinity norm
    # L-infinity norm between two vectors a and b is: max(|a_i - b_i|)
    # This matches MATLAB: pdist2(JS, JS, 'minkowski', inf)
    # Result shape: (N, N) where entry [i,j] is distance from point i to point j
    D_joint = cdist(JS, JS, metric="chebyshev")

    # For each point, find its k-th nearest neighbor distance
    # We sort distances and take index K (0-indexed), which is the K-th
    # neighbor EXCLUDING the point itself (which has distance 0 at index 0)
    # This matches MATLAB: pdist2(..., 'Smallest', K+1); epsilon = D(:, end)
    # np.sort sorts ascending, so index K gives the K-th neighbor distance
    D_joint_sorted = np.sort(D_joint, axis=1)

    # Column index K corresponds to the K-th neighbor
    # (index 0 is self, index 1 is 1st neighbor, ..., index K is K-th neighbor)
    epsilon = D_joint_sorted[:, K]  # Shape: (N,)

    # Step 2: Count neighbors in marginal spaces within epsilon
    # Compute pairwise distances in x-space only
    # This matches MATLAB: Dx = pdist2(x, x, 'minkowski', inf)
    D_x = cdist(x, x, metric="chebyshev")

    # For each point i, count how many points j satisfy D_x[i,j] < epsilon[i]
    # epsilon[:, None] reshapes epsilon from (N,) to (N,1) for broadcasting
    # This allows comparison of each row of D_x against its own epsilon value
    # We subtract 1 to exclude the point itself (distance 0 is always < epsilon)
    # This matches MATLAB: nx = sum(bsxfun(@lt, Dx, epsilon), 2) - 1
    nx = np.sum(D_x < epsilon[:, None], axis=1) - 1  # Shape: (N,)

    # Same for y-space
    # This matches MATLAB: ny = sum(bsxfun(@lt, Dy, epsilon), 2) - 1
    D_y = cdist(y, y, metric="chebyshev")
    ny = np.sum(D_y < epsilon[:, None], axis=1) - 1  # Shape: (N,)

    # Step 3: Apply KSG formula
    # digamma(K) and digamma(N) are scalar corrections
    # digamma(nx+1) and digamma(ny+1) are per-point corrections
    # The mean averages over all N points
    # This matches MATLAB: I = psi(K) + psi(n) - mean(psi(nx+1) + psi(ny+1))
    I = digamma(K) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))

    return float(I)


def cmi_knn(x, y, z, K=2):
    """
    Estimate Conditional Mutual Information I(X;Y|Z) using the KSG estimator.

    This is a translation of Dr. AlMomani's MATLAB cmiKnn function.

    The formula is:
        I(X;Y|Z) = psi(K) - mean(psi(nxz+1) + psi(nyz+1) - psi(nz+1))

    Notice: compared to MI, there is no psi(N) term, and we now work
    in the (x,z), (y,z), and z marginal spaces instead of x and y alone.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, dx)
    y : np.ndarray of shape (n_samples, dy)
    z : np.ndarray of shape (n_samples, dz)
        Conditioning variable.
    K : int
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.
    """
    # If z is empty, fall back to plain MI
    # This matches MATLAB: if isempty(z), I = miKnn(x,y,options)
    if z is None or z.size == 0:
        return mi_knn(x, y, K)

    # Ensure all inputs are 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    assert x.shape[0] == y.shape[0] == z.shape[0], (
        "x, y, z must have the same number of rows"
    )

    # Step 1: Build joint space (x,y,z) and find k-th neighbor distances
    # This matches MATLAB: JS = cat(2, x, y, z)
    JS = np.hstack([x, y, z])

    D_joint = cdist(JS, JS, metric="chebyshev")
    D_joint_sorted = np.sort(D_joint, axis=1)
    epsilon = D_joint_sorted[:, K]

    # Step 2: Count neighbors in (x,z), (y,z), and z marginal spaces
    # This matches MATLAB: Dxz = pdist2([x,z], [x,z], ...)
    D_xz = cdist(np.hstack([x, z]), np.hstack([x, z]), metric="chebyshev")
    nxz = np.sum(D_xz < epsilon[:, None], axis=1) - 1

    # This matches MATLAB: Dyz = pdist2([y,z], [y,z], ...)
    D_yz = cdist(np.hstack([y, z]), np.hstack([y, z]), metric="chebyshev")
    nyz = np.sum(D_yz < epsilon[:, None], axis=1) - 1

    # This matches MATLAB: Dz = pdist2(z, z, ...)
    D_z = cdist(z, z, metric="chebyshev")
    nz = np.sum(D_z < epsilon[:, None], axis=1) - 1

    # Step 3: Apply conditional KSG formula
    # This matches MATLAB: I = psi(K) - mean(psi(nxz+1)+psi(nyz+1)-psi(nz+1))
    I = digamma(K) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))

    return float(I)
