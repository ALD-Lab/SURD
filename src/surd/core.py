import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from itertools import combinations as icmb
from itertools import chain as ichain
from typing import Tuple, Dict
from . import it_tools as it
import warnings
try:
    import pymp  # optional: parallel loop in run_parallel (Linux and macOS)
except ImportError:
    pymp = None
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)



def _order(values: np.ndarray, sizes: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Indices that sort ``values`` in ascending order, with ties ordered by
    subset size (smaller first) and then by position.

    A subset and its superset often carry the same specific mutual
    information, for instance when the target depends on one source only.
    The superset must then come after the subset, so that the information
    is booked to the subset and not as synergy. ``np.argsort`` leaves the
    order of ties to the sorting algorithm, and round-off makes the two
    values differ in the last digits. Values closer than ``rtol`` times the
    largest magnitude are therefore treated as tied.
    """
    values = np.asarray(values, dtype=float)
    sizes = np.asarray(sizes)
    if values.size == 0:
        return np.arange(0)
    tol = rtol * max(np.abs(values).max(), 1.0)
    i0 = np.argsort(values, kind="stable")
    # Consecutive sorted values closer than tol share a group.
    group = np.empty(values.size, dtype=np.int64)
    group[i0] = np.concatenate(([0], np.cumsum(np.diff(values[i0]) > tol)))
    return np.lexsort((np.arange(values.size), sizes, group))

def assign(Is: Dict, p_t: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    """
    Assign the specific mutual information to redundant, unique and
    synergistic terms. This is the second half of :func:`surd`, separated so
    that the specific mutual information can come from any estimator, such
    as the histogram in :func:`surd` or the classifier in ``surd.neural``.

    Parameters:
    - Is (dict): Specific mutual information i(t; S_A) in bits, one array over
      the target states per non-empty subset A of the sources. Keys are
      tuples of 1-based source indices, ordered by subset size and then
      lexicographically, as ``itertools.combinations`` gives them.
    - p_t (np.ndarray): Probability of each target state.

    Returns:
    - I_R (dict): Redundancies and unique information for each variable combination.
    - I_S (dict): Synergies for each variable combination.
    - MI (dict): Mutual information for each variable combination.
    """
    p_t = np.atleast_1d(np.asarray(p_t, dtype=float))
    combs = list(Is.keys())
    Nvars = max(len(c) for c in combs)
    Nt = len(p_t)
    inds = range(1, Nvars + 1)

    # Compute mutual information for each combination of agent variables
    MI = {k: (Is[k] * p_t).sum() for k in Is.keys()}
    
    # Initialize redundancy and synergy terms
    I_R = {cc: 0 for cc in combs}
    I_S = {cc: 0 for cc in combs[Nvars:]}

    # Process each value of the target variable
    for t in range(Nt):
        # Extract specific mutual information for the current target value
        I1 = np.array([ii[t] for ii in Is.values()])
        
        # Sorting specific mutual information
        i1 = _order(I1, np.array([len(c) for c in combs]))
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        # Update specific mutual information based on existing maximum values
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l+1)[0]
            Il1max = I1[lens==l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        # Recompute sorting of updated specific mutual information values
        i1 = _order(I1, lens)
        lab = [lab[i_] for i_ in i1]
        
        # Compute differences in sorted specific mutual information values
        Di = np.diff(I1[i1], prepend=0.)
        red_vars = list(inds)

        # Distribute mutual information to redundancy and synergy terms
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_t[t]
            if len(ll) == 1:
                I_R[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    return I_R, I_S, MI


def surd(p: np.ndarray) -> Tuple[Dict, Dict, Dict, float]:
    '''
    Decompose the mutual information between a target variable and a set 
    of agent variables into three terms: Redundancy (I_R), Synergy (I_S), 
    and Unique (I_U) information.
    
    The surd function is designed to compute a decomposition of 
    the mutual information between a target variable T (signal in the future) 
    and agent variables A (signals in the present). This decomposition results 
    in terms related to redundancy (overlapping information), synergy 
    (information that arises only when considering multiple variables together),
    and unique information.
    
    Parameters:
    - p (np.ndarray): A multi-dimensional array of the histogram, where the first dimension 
      represents the target variable, and subsequent dimensions represent agent variables.
      
    Returns:
    - I_R (dict): Redundancies and unique information for each variable combination.
    - I_S (dict): Synergies for each variable combination.
    - MI (dict): Mutual information for each variable combination.
    - info_leak (float): Estimation of the information leak

    Example: To understand the mutual information between target variable T and 
    a combination of agent variables A1, A2, and A3, you can use:
    I_R, I_S, MI, info_leak = surd(p)
    '''

    # Ensure no zero values in the probability distribution to avoid NaNs during log computations
    p += 1e-14
    # Normalize the distribution
    p /= p.sum()

    # Total number of dimensions (target + agents)
    Ntot = p.ndim
    # Number of agent variables
    Nvars = Ntot - 1
    # Number of states for the target variable
    Nt = p.shape[0]
    inds = range(1, Ntot)

    # Calculation of information leak
    H  = it.entropy_nvars(p, (0,) )
    Hc = it.cond_entropy(p, (0,), range(1,Ntot) )
    info_leak = Hc/H

    # Compute the marginal distribution of the target variable
    p_s = p.sum(axis=(*inds,), keepdims=True)

    # Prepare for specific mutual information computation
    combs, Is = [], {}
    
    # Iterate over all combinations of agent variables
    for i in inds:
        for j in list(icmb(inds, i)):
            combs.append(j)
            noj = tuple(set(inds) - set(j))
            
            # Compute joint and conditional distributions for current combinations
            p_a = p.sum(axis=(0, *noj), keepdims=True)
            p_as = p.sum(axis=noj, keepdims=True)

            p_a_s = p_as / p_s
            p_s_a = p_as / p_a

            # Compute specific mutual information
            Is[j] = (p_a_s * (it.mylog(p_s_a) - it.mylog(p_s))).sum(axis=j).ravel()

    # Assign the specific mutual information to redundant, unique and
    # synergistic terms
    I_R, I_S, MI = assign(Is, p_s.squeeze())

    return I_R, I_S, MI, info_leak


def surd_hd(Y: np.ndarray, nbins, max_combs) -> Tuple[Dict, Dict, Dict]:
    '''
    Extension of surd to high-dimensional systems. It computes the 
    the decomposition of information up to a given number of maximum combination
    between variables.
    
    Parameters:
    - Y (np.ndarray): A multi-dimensional array with the temporal evolution of the variables. 
    The first dimension represents the target variable, and subsequent dimensions represent 
    agent variables.
    - nbins: Number of bins to discretize the histogram.
    - max_combs: maximum order of combitations for synergistic contributions
      
    Returns:
    - I_R (dict): Redundancies and unique information for each variable combination.
    - I_S (dict): Synergies for each variable combination.
    - MI (dict): Mutual information for each variable combination.

    Example: To understand the mutual information between target variable T and 
    a combination of agent variables A1, A2, and A3, you can use:
    I_R, I_S, MI = surd(p)
    '''

    # Total number of dimensions (target + agents)
    Ntot = Y.shape[0]
    # Number of agent variables
    Nvars = Ntot - 1
    # Limit the maximum number of combinations to max_combs
    max_inds = range(1, max_combs+1)
    tot_inds = range(1, Ntot)

    # Compute the marginal distribution of the target variable
    p_target = it.myhistogram(Y[0,:].T, nbins)
    p_target = p_target.reshape((nbins,) + (1,) * (Ntot - 1))

    # Prepare for specific mutual information computation
    combs, Is = [], {}
    red_combs= []
    
    # Iterate over all combinations of agent variables
    for i in max_inds:
        for j in list(icmb(tot_inds, i)):
                combs.append(j)
                # noj = tuple(set(inds) - set(j))

                shape = np.ones(Ntot, dtype=int)
            
                # Compute joint distributions for current combinations
                p_a = it.myhistogram(Y[j,:].T, nbins)
                for index in j:
                    shape[index] = nbins
                p_a = p_a.reshape(tuple(shape))
                p_as = it.myhistogram(Y[(0,) + j,:].T, nbins)
                shape[0] = nbins
                p_as = p_as.reshape(tuple(shape))

                # Compute conditional distributions for current combinations
                p_a_s = p_as / p_target
                p_s_a = p_as / p_a

                # Compute specific mutual information
                Is[j] = (p_a_s * (it.mylog(p_s_a) - it.mylog(p_target))).sum(axis=j).ravel()

    # Compute mutual information for each combination of agent variables
    MI = {k: (Is[k] * p_target.squeeze()).sum() for k in Is.keys()}
    
    # Initialize redundancy and synergy terms
    for i in tot_inds:
        for j in list(icmb(tot_inds, i)):
            red_combs.append(j)
    I_R = {cc: 0 for cc in red_combs}
    I_S = {cc: 0 for cc in combs[Nvars:]}

    # Process each value of the target variable
    for t in range(nbins):
        # Extract specific mutual information for the current target value
        I1 = np.array([ii[t] for ii in Is.values()])
        
        # Sorting specific mutual information
        i1 = _order(I1, np.array([len(c) for c in combs]))
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        # Update specific mutual information based on existing maximum values
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l+1)[0]
            Il1max = I1[lens==l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        # Recompute sorting of updated specific mutual information values
        i1 = _order(I1, lens)
        lab = [lab[i_] for i_ in i1]
        
        # Compute differences in sorted specific mutual information values
        Di = np.diff(I1[i1], prepend=0.)
        red_vars = list(tot_inds)

        # Distribute mutual information to redundancy and synergy terms
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_target.squeeze()[t]
            if len(ll) == 1:
                I_R[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    return I_R, I_S, MI


def plot(I_R, I_S, info_leak, axs, nvars, threshold=0):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors['redundant'] = mcolors.to_rgb('#003049')
    colors['unique'] = mcolors.to_rgb('#d62828')
    colors['synergistic'] = mcolors.to_rgb('#f77f00')

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1-c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = 'U' if len(comb) == 1 else 'R'
            I_R_keys.append(prefix + ''.join(map(str, comb)))
            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")
    
    # Synergestic Contributions
    I_S_keys = ['S' + ''.join(map(str, comb)) for r in range(2, nvars+1) for comb in icmb(range(1, nvars + 1), r)]
    I_S_labels = [f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$" for r in range(2, nvars+1) for comb in icmb(range(1, nvars + 1), r)]

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [I_R.get(tuple(map(int, key[1:])), 0) if 'U' in key or 'R' in key 
          else I_S.get(tuple(map(int, key[1:])), 0) 
          for key in label_keys]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]
    
    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if 'U' in label:
            color = colors['unique']
        elif 'S' in label:
            color = colors['synergistic']
        else:
            color = colors['redundant']
        axs[0].bar(label, value, color=color, edgecolor='black',linewidth=1.5)

    if nvars == 2:
        axs[0].set_box_aspect(1/2.5)
    else:
        axs[0].set_box_aspect(1/4)

    # Plotting the information leak bar
    axs[1].bar(' ', info_leak, color='gray', edgecolor='black')
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0., 1.])
    axs[0].set_ylim([0., 1.])

    # change all spines
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def plot_nlabels(I_R, I_S, info_leak, axs, nvars, nlabels=-1):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors['redundant'] = mcolors.to_rgb('#003049')
    colors['unique'] = mcolors.to_rgb('#d62828')
    colors['synergistic'] = mcolors.to_rgb('#f77f00')

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1-c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = 'U' if len(comb) == 1 else 'R'
            I_R_keys.append(prefix + ''.join(map(str, comb)))
            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")
    
    # Synergestic Contributions
    I_S_keys = ['S' + ''.join(map(str, comb)) for r in range(2, nvars+1) for comb in icmb(range(1, nvars + 1), r)]
    I_S_labels = [f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$" for r in range(2, nvars+1) for comb in icmb(range(1, nvars + 1), r)]

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [I_R.get(tuple(map(int, key[1:])), 0) if 'U' in key or 'R' in key 
          else I_S.get(tuple(map(int, key[1:])), 0) 
          for key in label_keys]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    top_n_indices = np.argsort(values)[-nlabels:]

    # Filter both the values and labels arrays
    filtered_values = values[top_n_indices]
    filtered_labels = np.array(labels)[top_n_indices]
    original_order_indices = np.argsort(top_n_indices)
    filtered_values_in_original_order = filtered_values[original_order_indices]
    filtered_labels_in_original_order = filtered_labels[original_order_indices]

    # Convert filtered arrays back to lists if necessary
    values = filtered_values_in_original_order
    labels = filtered_labels_in_original_order.tolist()
    
    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if 'U' in label:
            color = colors['unique']
        elif 'S' in label:
            color = colors['synergistic']
        else:
            color = colors['redundant']
        axs[0].bar(label, value, color=color, edgecolor='black',linewidth=1.5)

    axs[0].set_box_aspect(1/4)

    # Plotting the information leak bar
    axs[1].bar(' ', info_leak, color='gray', edgecolor='black')
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0., 1.])
    axs[0].set_ylim([0., 1.])

    # change all spines
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def nice_print( r_, s_, mi_, leak_ ):
    '''Print the normalized redundancies, unique and synergy particles'''

    r_ = {key: value / max(mi_.values()) for key, value in r_.items()}
    s_ = {key: value / max(mi_.values()) for key, value in s_.items()}

    print( '    Redundant (R):' )
    for k_, v_ in r_.items():
        if len(k_) > 1:
            print( f'        {str(k_):12s}: {v_:5.4f}' )

    print( '    Unique (U):' )
    for k_, v_ in r_.items():
        if len(k_) == 1:
            print( f'        {str(k_):12s}: {v_:5.4f}' )

    print( '    Synergystic (S):' )
    for k_, v_ in s_.items():
        print( f'        {str(k_):12s}: {v_:5.4f}' )

    print(f'    Information Leak: {leak_ * 100:5.2f}%')


def run(X, nvars, nlag, nbins, axs):

    information_flux = {}

    for i in range(nvars):
        print(f'SURD CAUSALITY FOR SIGNAL {i+1}')

        # Organize data (0 target variable, 1: agent variables)
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])

        # Run SURD
        hist, _ = np.histogramdd(Y.T, nbins)
        I_R, I_S, MI, info_leak = surd(hist)
        
        # Print results
        nice_print(I_R, I_S, MI, info_leak)

        # Plot SURD
        information_flux[i+1] = plot(I_R, I_S, info_leak, axs[i,:], nvars, threshold=-0.01)
        
        # Plot formatting
        axs[i,0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$', pad=12)
        axs[i,1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$', pad=20)
        axs[i,0].set_xticklabels(axs[i,0].get_xticklabels(), fontsize=20, rotation = 60, ha = 'right', rotation_mode='anchor')
        print('\n')

    # Show the results
    for i in range(0,nvars-1):
        axs[i,0].set_xticklabels('')

    return I_R, I_S, MI, info_leak


def run_parallel(X, nvars, nlag, nbins, axs):

    information_flux = {}

    def _one(i):
        # Organize data (0 target variable, 1: agent variables)
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])

        # Run SURD
        hist, _ = np.histogramdd(Y.T, nbins)
        I_R, I_S, MI, info_leak = surd(hist)

        # Print results
        print(f'SURD CAUSALITY FOR SIGNAL {i+1}')
        nice_print(I_R, I_S, MI, info_leak)
        print('\n')
        return I_R, I_S, MI, info_leak

    if pymp is not None:
        Rd_results = pymp.shared.dict({})  # Dictionary to store redundant contributions
        Sy_results = pymp.shared.dict({})  # Dictionary to store synergistic contributions
        MI_results = pymp.shared.dict({})   # Dictionary to store mutual information results
        info_leak_results = pymp.shared.dict({})  # Dictionary to store information leak results
        with pymp.Parallel(nvars) as par:
            for i in par.range(nvars):
                Rd_results[i+1], Sy_results[i+1], MI_results[i+1], info_leak_results[i+1] = _one(i)
    else:  # pymp not installed (e.g. Windows): same computation, one target at a time
        Rd_results, Sy_results, MI_results, info_leak_results = {}, {}, {}, {}
        for i in range(nvars):
            Rd_results[i+1], Sy_results[i+1], MI_results[i+1], info_leak_results[i+1] = _one(i)

    for i in range(nvars):
        # Plot SURD
        information_flux[i+1] = plot(Rd_results[i+1], Sy_results[i+1], info_leak_results[i+1], axs[i,:], nvars, threshold=-0.01)
        
        # Plot formatting
        axs[i,0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$', pad=12)
        axs[i,1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$', pad=20)
        axs[i,0].set_xticklabels(axs[i,0].get_xticklabels(), fontsize=20, rotation = 60, ha = 'right', rotation_mode='anchor')

    # Show the results
    for i in range(0,nvars-1):
        axs[i,0].set_xticklabels('')

    # Same return value as before: the decomposition for the first target
    return Rd_results[1], Sy_results[1], MI_results[1], info_leak_results[1]


def plot_multiple_lags(I_R, I_S, info_leak, axs, n_vars_lag, n_lag, threshold=0):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axis for plotting
    :param n_vars_lag: Number of variables including lags
    :param n_lag: Number of lags
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors['redundant'] = mcolors.to_rgb('#003049')
    colors['unique'] = mcolors.to_rgb('#d62828')
    colors['synergistic'] = mcolors.to_rgb('#f77f00')

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1-c) * 0.4 for c in rgb])

    # Generate keys and labels
    n_vars = n_vars_lag // n_lag

    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    # for r in range(1, n_vars_lag + 1):
    for r in range(n_vars_lag, 0, -1):
        for comb in itertools.combinations(range(1, n_vars_lag + 1), r):
            prefix = 'U' if len(comb) == 1 else 'R'
            I_R_keys.append(prefix + ''.join(map(str, comb)))

            # New label generation with subscripts for lags
            new_comb_labels = []
            for c in comb:
                lag_number = (c - 1) // n_vars
                var_number = (c - 1) % n_vars + 1
                new_label = f"{var_number}_{{{lag_number+1}}}"
                new_comb_labels.append(new_label)

            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(new_comb_labels)}}}$")
    
    # Synergestic Contributions
    I_S_keys = []
    I_S_labels = []
    for r in range(2, n_vars_lag + 1):  # Starting from 2 because synergistic contributions require at least two variables
        for comb in itertools.combinations(range(1, n_vars_lag + 1), r):
            # Generating the key
            I_S_keys.append('S' + ''.join(map(str, comb)))

            # Generating the label with subscripts for lags
            new_comb_labels = []
            for c in comb:
                lag_number = (c - 1) // n_vars
                var_number = (c - 1) % n_vars + 1
                new_label = f"{var_number}_{{{lag_number+1}}}"
                new_comb_labels.append(new_label)

            I_S_labels.append(f"$\\mathrm{{S}}{{{''.join(new_comb_labels)}}}$")

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [I_R.get(tuple(map(int, key[1:])), 0) if 'U' in key or 'R' in key 
          else I_S.get(tuple(map(int, key[1:])), 0) 
          for key in label_keys]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if 'U' in label:
            color = colors['unique']
        elif 'S' in label:
            color = colors['synergistic']
        else:
            color = colors['redundant']
        axs[0].bar(label, value, color=color, edgecolor='black',linewidth=1.5)
    
    # Plotting the bar graph of information measures
    axs[0].set_xticks(range(len(values)))
    shift_labels = axs[0].set_xticklabels(labels, fontsize=15, rotation = 60, ha = 'right', rotation_mode='anchor')
    axs[0].set_box_aspect(1/5)

    # Plotting the information leak bar
    axs[1].bar(' ', info_leak, color='gray', edgecolor='black')
    axs[1].set_ylim([0, 1])

    # change all spines
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(1.5)
        axs[1].spines[axis].set_linewidth(1.5)

    # increase tick width
    axs[0].tick_params(width=1.5)
    axs[1].tick_params(width=1.5)

    return dict(zip(label_keys, values))


def run_multiple_lags(X, nvars, nlag, nbins, max_combs, axs):
    "Run SURD causality for different lags (from lag 1 up to nlag)"
    information_flux = {}

    for i in range(nvars):
        print(f'SURD CAUSALITY FOR SIGNAL {i+1}')

        # Organize data (0 target variable, 1: agent variables)
        Y = X[i, nlag+1:]
        # Create the lagged versions of X and append to the list
        for lag in range(nlag, 0, -1):
            Y = np.vstack([Y, X[:, lag:-nlag + lag - 1]])

        # Run SURD
        I_R, I_S, MI = surd_hd(Y, nbins, max_combs)

        # Calculate information leak
        hist = it.myhistogram(Y[0,:].T, nbins)
        H  = it.entropy_nvars(hist, (0,) )
        info_leak = 1 - (sum(I_R.values()) + sum(I_S.values())) / H
        
        # Print results
        nice_print(I_R, I_S, MI, info_leak)
        print('\n')

        # Plot SURD
        information_flux[i+1] = plot_multiple_lags(I_R, I_S, info_leak, axs[i,:], nvars*nlag, nlag, threshold=-0.01)
        
        # Plot formatting
        axs[i,0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$', pad=10)
        axs[i,1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$', pad=18)
        axs[i,1].set_yticks([0,1])
        axs[i,0].set_xticklabels(axs[i,0].get_xticklabels(), fontsize=14, rotation = 60, ha = 'right', rotation_mode='anchor')

        # change all spines
        for axis in ['top','bottom','left','right']:
            axs[i,0].spines[axis].set_linewidth(2.5)
            axs[i,1].spines[axis].set_linewidth(2.5)
        axs[i,0].set_box_aspect(1/4.5)

    # Show the results
    for i in range(0,nvars-1):
        axs[i,0].set_xticklabels('')

    return I_R, I_S, MI, info_leak