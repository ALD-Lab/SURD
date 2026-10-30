import numpy as np


def myhistogram(x, nbins):

    hist, _ = np.histogramdd(x, nbins)

    hist += 1e-14
    hist /= hist.sum()

    return hist


def mylog(x):
    """
    Compute the logarithm in base 2 avoiding singularities.
    
    Parameters:
    - x (np.array): Input data.

    Returns:
    - np.array: Logarithm in base 2 of the input.
    """
    valid_indices = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))
    
    log_values = np.zeros_like(x)
    log_values[valid_indices] = np.log2(x[valid_indices])
    
    return log_values


def entropy(p):
    """
    Compute the entropy of a discrete probability distribution function.

    Parameters:
    - p (np.array): Probability distribution of the signal.

    Returns:
    - float: Entropy of the given distribution.
    """
    return -np.sum(p * mylog(p))


def entropy_nvars(p, indices):
    """
    Compute the joint entropy for specific dimensions of a probability distribution.

    Parameters:
    - p (np.array): N-dimensional joint probability distribution.
    - indices (tuple): Dimensions over which the entropy is to be computed.

    Returns:
    - float: Joint entropy for specified dimensions.

    Example: compute the joint entropy H(X0,X3,X7)
    >>> entropy_nvars(p, (0,3,7))
    """
    excluded_indices = tuple(set(range(p.ndim)) - set(indices))
    marginalized_distribution = p.sum(axis=excluded_indices)

    return entropy(marginalized_distribution)


def cond_entropy(p, target_indices, conditioning_indices):
    """
    Compute the conditional entropy between two sets of variables.

    Parameters:
    - p (np.array): N-dimensional joint probability distribution.
    - target_indices (tuple): Variables for which entropy is to be computed.
    - conditioning_indices (tuple): Conditioning variables.

    Returns:
    - float: Conditional entropy.

    Example: compute the conditional entropy H(X0,X2|X7)
    >>> cond_entropy(p, (0, 2), (7,))
    """
    joint_entropy = entropy_nvars(p, set(target_indices) | set(conditioning_indices))
    conditioning_entropy = entropy_nvars(p, conditioning_indices)

    return joint_entropy - conditioning_entropy


def mutual_info(p, set1_indices, set2_indices):
    """
    Compute the mutual information between two sets of variables.

    Parameters:
    - p (np.array): N-dimensional joint probability distribution.
    - set1_indices (tuple): Indices of the first set of variables.
    - set2_indices (tuple): Indices of the second set of variables.

    Returns:
    - float: Mutual information.

    Example: compute the mutual information I(X0,X5;X4,X2)
    >>> mutual_info(p, (0, 5), (4, 2))
    """
    entropy_set1 = entropy_nvars(p, set1_indices)
    conditional_entropy = cond_entropy(p, set1_indices, set2_indices)

    return entropy_set1 - conditional_entropy


def cond_mutual_info(p, ind1, ind2, ind3):
    """
    Compute the conditional mutual information between two sets of variables 
    conditioned to a third set.

    Parameters:
    - p (np.array): N-dimensional joint probability distribution.
    - ind1 (tuple): Indices of the first set of variables.
    - ind2 (tuple): Indices of the second set of variables.
    - ind3 (tuple): Indices of the conditioning variables.

    Returns:
    - float: Conditional mutual information.

    Example: compute the conditional mutual information I(X0,X5;X4,X2|X1)
    cond_mutual_info(p, (0, 5), (4, 2), (1,)))
    """
    # Merge indices of ind2 and ind3
    combined_indices = tuple(set(ind2) | set(ind3))
    
    # Compute conditional mutual information
    return cond_entropy(p, ind1, ind3) - cond_entropy(p, ind1, combined_indices)


def transfer_entropy(p, target_var):
    """
    Calculate the transfer entropy from each input variable to the target variable.

    Parameters:
    - p (np.array): Multi-dimensional array containing the pdfs of the variables.
      The first dimension corresponds to the index of the variable:
          p[0]  -> target variable (in future)
          p[1:] -> input variables (at present time)

    Returns:
    - np.array: Transfer entropy values for each input variable.
    """
    num_vars = len(p.shape) - 1  # Excluding the future variable
    TE = np.zeros(num_vars)
    
    for i in range(1, num_vars + 1):
        # The indices for the present variables
        present_indices = tuple(range(1, num_vars + 1))
        
        # The indices for the present variables excluding the i-th variable
        # conditioning_indices = tuple([target_var] + [j for j in range(1, num_vars + 1) if j != i])
        conditioning_indices = tuple([target_var] + [j for j in range(1, num_vars + 1) if j != i and j != target_var])
        
        # Conditional entropy of the future state of the target variable given its own past
        cond_ent_target_given_past = cond_entropy(p, (0,), conditioning_indices)
        
        # Conditional entropy of the future state of the target variable given its own past and the ith input variable
        cond_ent_target_given_past_and_input = cond_entropy(p, (0,), present_indices)
        
        # Transfer entropy calculation
        TE[i-1] = cond_ent_target_given_past - cond_ent_target_given_past_and_input
    
    return TE


