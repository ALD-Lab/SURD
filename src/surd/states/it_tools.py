import math
import numpy as np
from itertools import combinations as icmb
from itertools import chain as ichain


def myhistogram(x, nbins):

    hist, _ = np.histogramdd(x, nbins)

    hist += 1e-14
    hist /= hist.sum()

    return hist


def mylog(x):
    """
    Compute the logarithm in base 2 avoiding singularities.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        Logarithm in base 2 of the input.
    """
    valid_indices = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))

    log_values = np.zeros_like(x)
    log_values[valid_indices] = np.log2(x[valid_indices])

    return log_values


def safe_div(num, den):
    """
    Element-wise division that returns 0 where the denominator is 0.

    Conditional probabilities of the form p(x|y) are undefined when p(y) = 0;
    by the standard 0 * log(0) = 0 convention, treating the ratio as 0 at
    those entries yields the same result as the unobserved state contributing
    nothing — without the NaN/inf propagation produced by naive division.

    Parameters
    ----------
    num, den : np.ndarray
        Numerator and denominator, broadcastable to a common shape.

    Returns
    -------
    np.ndarray
        Element-wise num / den, with 0 wherever den == 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, 0.0)


def entropy(p):
    """
    Compute the entropy of a discrete probability distribution function.

    Parameters
    ----------
    p : np.ndarray
        Probability distribution of the signal.

    Returns
    -------
    float
        Entropy of the given distribution.
    """
    return -np.sum(p * mylog(p))


def entropy_nvars(p, indices):
    """
    Compute the joint entropy for specific dimensions of a probability distribution.

    Parameters
    ----------
    p : np.ndarray
        N-dimensional joint probability distribution.
    indices : tuple
        Dimensions over which the entropy is to be computed.

    Returns
    -------
    float
        Joint entropy for specified dimensions.

    Examples
    --------
    Compute the joint entropy H(X0,X3,X7):

    >>> entropy_nvars(p, (0, 3, 7))
    """
    excluded_indices = tuple(set(range(p.ndim)) - set(indices))
    marginalized_distribution = p.sum(axis=excluded_indices)

    return entropy(marginalized_distribution)


def cond_entropy(p, target_indices, conditioning_indices):
    """
    Compute the conditional entropy between two sets of variables.

    Parameters
    ----------
    p : np.ndarray
        N-dimensional joint probability distribution.
    target_indices : tuple
        Variables for which entropy is to be computed.
    conditioning_indices : tuple
        Conditioning variables.

    Returns
    -------
    float
        Conditional entropy.

    Examples
    --------
    Compute the conditional entropy H(X0,X2|X7):

    >>> cond_entropy(p, (0, 2), (7,))
    """
    joint_entropy = entropy_nvars(p, set(target_indices) | set(conditioning_indices))
    conditioning_entropy = entropy_nvars(p, conditioning_indices)

    return joint_entropy - conditioning_entropy


def mutual_info(p, set1_indices, set2_indices):
    """
    Compute the mutual information between two sets of variables.

    Parameters
    ----------
    p : np.ndarray
        N-dimensional joint probability distribution.
    set1_indices : tuple
        Indices of the first set of variables.
    set2_indices : tuple
        Indices of the second set of variables.

    Returns
    -------
    float
        Mutual information.

    Examples
    --------
    Compute the mutual information I(X0,X5;X4,X2):

    >>> mutual_info(p, (0, 5), (4, 2))
    """
    entropy_set1 = entropy_nvars(p, set1_indices)
    conditional_entropy = cond_entropy(p, set1_indices, set2_indices)

    return entropy_set1 - conditional_entropy


def cond_mutual_info(p, ind1, ind2, ind3):
    """
    Compute the conditional mutual information between two sets of variables
    conditioned to a third set.

    Parameters
    ----------
    p : np.ndarray
        N-dimensional joint probability distribution.
    ind1 : tuple
        Indices of the first set of variables.
    ind2 : tuple
        Indices of the second set of variables.
    ind3 : tuple
        Indices of the conditioning variables.

    Returns
    -------
    float
        Conditional mutual information.

    Examples
    --------
    Compute the conditional mutual information I(X0,X5;X4,X2|X1):

    >>> cond_mutual_info(p, (0, 5), (4, 2), (1,))
    """
    # Merge indices of ind2 and ind3
    combined_indices = tuple(set(ind2) | set(ind3))

    # Compute conditional mutual information
    return cond_entropy(p, ind1, ind3) - cond_entropy(p, ind1, combined_indices)


def information_flux(p):
    """
    Compute the information flux from N input variables to a target variable.

    This function computes all possible fluxes combinations of the N input
    variables to the target variable, given the joint pdf of N+1 variables.
    It evaluates the conditional entropies to ascertain the flux of information
    from various combinations of input variables to the target variable.

    Parameters
    ----------
    p : np.ndarray
        Multi-dimensional array containing the pdfs of the variables.
        The first dimension corresponds to the index of the variable:

        - p[0]  : target variable (in future)
        - p[1:] : input variables (at present time)

    Returns
    -------
    dict
        Dictionary containing all possible fluxes from input variables
        to the target variable. The key represents a tuple of input
        variable indices contributing to the flux, and the associated
        value represents the magnitude of the flux from those input
        variables to the target variable.

    Notes
    -----
    The computed fluxes represent the influence or the contribution
    of different combinations of input variables on the target
    variable. They are beneficial in analyzing multivariate systems
    to understand the interactions and dependencies between variables.

    Examples
    --------
    If p is a multi-dimensional array containing the pdfs of the
    variables, information_flux(p) will return a dictionary T, where each
    key is a tuple representing a combination of input variables, and
    the corresponding value represents the information flux from those
    variables to the target variable.
    """

    Np = len(p.shape)  # Number of dimensions in p
    inds = range(1, Np)  # Indices representing input variables
    p /= p.sum()

    T = {}  # Initialize the dictionary to hold the computed fluxes

    for i in inds:
        for j in list(icmb(inds, i)):
            noi = tuple(set(inds) - set(j))
            Hc_j_noi = cond_entropy(
                p, (0,), noi
            )  # Conditional entropy with respect to the non-involved variables
            Hc_j_all = cond_entropy(
                p, (0,), inds
            )  # Conditional entropy with respect to all input variables
            T[j] = Hc_j_noi - Hc_j_all  # Compute the flux

    for i in inds:
        for j in list(icmb(inds, i)):
            # Determine the combinations that need to be subtracted from T[j]
            lj = [list(icmb(j, k)) for k in range(len(j))][1:]
            T[j] -= sum(
                [T[a] for a in list(ichain.from_iterable(lj))]
            )  # Subtract the computed fluxes to get the final flux for the combination j

    return T  # Return the computed fluxes


def transfer_entropy(p, target_var):
    """
    Calculate the transfer entropy from each input variable to the target variable.

    Parameters
    ----------
    p : np.ndarray
        Multi-dimensional array containing the pdfs of the variables.
        The first dimension corresponds to the index of the variable:

        - p[0]  : target variable (in future)
        - p[1:] : input variables (at present time)
    target_var : int
        Index of the target variable.

    Returns
    -------
    np.ndarray
        Transfer entropy values for each input variable.
    """
    num_vars = len(p.shape) - 1  # Excluding the future variable
    TE = np.zeros(num_vars)

    for i in range(1, num_vars + 1):
        # The indices for the present variables
        present_indices = tuple(range(1, num_vars + 1))

        # The indices for the present variables excluding the i-th variable
        # conditioning_indices = tuple([target_var] + [j for j in range(1, num_vars + 1) if j != i])
        conditioning_indices = tuple(
            [target_var]
            + [j for j in range(1, num_vars + 1) if j != i and j != target_var]
        )

        # Conditional entropy of the future state of the target variable given its own past
        cond_ent_target_given_past = cond_entropy(p, (0,), conditioning_indices)

        # Conditional entropy of the future state of the target variable given its own past and the ith input variable
        cond_ent_target_given_past_and_input = cond_entropy(p, (0,), present_indices)

        # Transfer entropy calculation
        TE[i - 1] = cond_ent_target_given_past - cond_ent_target_given_past_and_input

    return TE


def random_permutation(x):
    """Randomly permute an array."""
    N = x.shape[0]
    permuted_indices = np.random.permutation(N)
    return x[permuted_indices]


def hist_knn(Y, bins=10, k=5):
    """
    Estimate the probability density function of Y using k-nearest neighbors.

    Returns a histogram of the estimated densities at the bin centers for each
    dimension.

    Parameters
    ----------
    Y : np.ndarray
        Input data array with shape (n_samples, n_features).
    bins : int, optional
        The number of bins for each dimension (default is 10).
    k : int, optional
        Number of nearest neighbors to use for density estimation (default is 5).

    Returns
    -------
    pdf_grid : np.ndarray
        Grid of estimated densities at the bin centers.
    bin_centers : list of np.ndarray
        Centers of the bins for each dimension.
    """
    # scikit-learn is only needed here (optional extra ``surd[states]``).
    from sklearn.neighbors import NearestNeighbors

    # Compute bin edges and centers for each dimension
    max_abs = np.percentile(Y, 99.99)
    max_abs = np.floor(max_abs)
    bin_width = 2 * max_abs / (bins - 1)  # Calculate the bin width

    bin_edges = [
        np.linspace(-max_abs, max_abs + bin_width, bins + 1)
        for dim in range(Y.shape[1])
    ]
    # bin_edges = [np.linspace(np.min(Y[:, dim]), np.max(Y[:, dim]), bins + 1) for dim in range(Y.shape[1])]
    bin_centers = [0.5 * (edges[:-1] + edges[1:]) for edges in bin_edges]

    # Create a grid of all possible combinations of bin centers
    bin_centers_mesh = np.array(np.meshgrid(*bin_centers, indexing="ij"))

    # Reshape the meshgrid to pass to NearestNeighbors
    bin_centers_flat = bin_centers_mesh.reshape(Y.shape[1], -1).T

    # Initialize NearestNeighbors and fit to data
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Y)

    # Find distances to the k-th nearest neighbors at each bin center
    distances, _ = nn.kneighbors(bin_centers_flat)

    # Get the distance to the k-th nearest neighbor
    kth_distances = distances[:, k - 1]

    # Compute the volume of the n-dimensional sphere with radius equal to the distance
    n_dimensions = Y.shape[1]
    volume_const = (np.pi ** (n_dimensions / 2)) / math.gamma(n_dimensions / 2 + 1)
    volumes = volume_const * (kth_distances**n_dimensions)

    # Estimate density at each bin center
    pdf = k / (Y.shape[0] * volumes)

    # Reshape the PDF to the grid shape
    pdf_grid = pdf.reshape([bins] * n_dimensions)

    return pdf_grid, bin_centers


def generate_loops(Nvars, Nbins):
    """
    Generate nested loop code as a string.

    Parameters
    ----------
    Nvars : int
        Number of variables (loop depth).
    Nbins : int
        Range for each loop variable.

    Returns
    -------
    str
        String containing the generated loop code.
    """
    loop_code = ""
    indent = ""
    vars_list = [
        chr(97 + i) for i in range(Nvars)
    ]  # Generate variable names: t, u, v, ...

    # Create nested for loops
    for var in vars_list:
        loop_code += f"{indent}for {var} in range({Nbins}):\n"
        indent += "    "  # Increase indentation

    # Add body of the innermost loop
    vars_sum = " + ".join(vars_list)
    loop_code += f"{indent}print('Sum of {', '.join(vars_list)}:', {vars_sum})\n"

    return loop_code
