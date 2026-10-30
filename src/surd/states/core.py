import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from itertools import combinations as icmb
from itertools import chain as ichain
from typing import Tuple, Dict
from . import it_tools as it
import warnings
import heapq
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

my_colors = {}
my_colors["redundant"] = mcolors.to_rgb("#003049")
my_colors["unique"] = mcolors.to_rgb("#d62828")
my_colors["synergistic"] = mcolors.to_rgb("#f77f00")
my_colors["red"] = mcolors.to_rgb("#d62828")
my_colors["green"] = mcolors.to_rgb("#6ca13b")
my_gray = mcolors.to_rgb("#000000")
my_gray = tuple([c + (1 - c) * 0.8 for c in my_gray])
my_colors["gray"] = my_gray

for key, value in my_colors.items():
    rgb = mcolors.to_rgb(value)
    my_colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])


def surd_states(p: np.ndarray) -> Tuple[Dict, Dict, Dict, float, float]:
    """
    Decompose the mutual information between a target variable and agent variables.

    This function decomposes the mutual information between a target variable T
    (signal in the future) and agent variables A (signals in the present) into
    three terms: Redundancy (I_R), Synergy (I_S), and Unique (I_U) information.

    Parameters
    ----------
    p : np.ndarray
        A multi-dimensional array of the histogram, where the first dimension
        represents the target variable, and subsequent dimensions represent
        observable variables.

    Returns
    -------
    I_R : dict
        Redundancies and unique information for each variable combination.
    I_S : dict
        Synergies for each variable combination.
    MI : dict
        Mutual information for each variable combination.
    info_leak : float
        Estimation of the information leak
    Rd_states : dict
        Redundancy states
    Un_states : dict
        Unique information states
    Sy_states : dict
        Synergy states

    Examples
    --------
    To understand the mutual information between target variable T and
    a combination of agent variables A1, A2, and A3:

    >>> I_R, I_S, MI, info_leak, Rd_states, Un_states, Sy_states = surd_states(p)
    """
    # Normalize the distribution. Zero-valued entries are handled downstream
    # by ``it.mylog`` (returns 0 for 0/NaN/inf) and ``it.safe_div`` (returns 0
    # when the denominator is 0), so no additive regularization is needed.
    p = p / p.sum()

    # Total number of dimensions (target + agents)
    Ntot = p.ndim
    # Number of agent variables
    Nvars = Ntot - 1
    # Number of states for the target variable
    Nt = p.shape[0]
    inds = range(1, Ntot)

    # Calculation of information leak
    H = it.entropy_nvars(p, (0,))
    Hc = it.cond_entropy(p, (0,), range(1, Ntot))
    info_leak = Hc / H

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

            p_a_s = it.safe_div(p_as, p_s)
            p_s_a = it.safe_div(p_as, p_a)

            # Compute specific mutual information
            Is[j] = (p_a_s * (it.mylog(p_s_a) - it.mylog(p_s))).sum(axis=j).ravel()

    # Compute mutual information for each combination of agent variables
    MI = {k: (Is[k] * p_s.squeeze()).sum() for k in Is.keys()}

    # Initialize redundancy and synergy terms
    I_R = {cc: 0 for cc in combs}
    I_S = {cc: 0 for cc in combs[Nvars:]}

    # Specific unique contributions from one state of one variable to one state of the target variable
    # Un_states: dict[agent] that contains an array of size (Nt, Nbins_agent)
    Un_states = {(agent,): np.zeros((Nt, p.shape[agent])) for agent in inds}
    Sy_states = {}
    Rd_states = {}
    for i, agenti in enumerate(inds):
        for agentj in inds[i + 1 :]:
            Sy_states[(agenti, agentj)] = np.zeros(
                (Nt, p.shape[agenti], p.shape[agentj])
            )
            Rd_states[(agenti, agentj)] = np.zeros(
                (Nt, p.shape[agenti], p.shape[agentj])
            )
    I_R_sp = {cc: np.zeros((Nt)) for cc in combs}
    I_S_sp = {cc: np.zeros((Nt)) for cc in combs[Nvars:]}

    # Process each value of the target variable
    for t in range(Nt):
        # Extract specific mutual information for the current target value
        I1 = np.array([ii[t] for ii in Is.values()])

        # Sorting specific mutual information
        i1 = np.argsort(I1)
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        # Update specific mutual information based on existing maximum values
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l + 1)[0]
            Il1max = I1[lens == l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        # Recompute sorting of updated specific mutual information values
        i1 = np.argsort(I1)
        lab = [lab[i_] for i_ in i1]

        # Compute differences in sorted specific mutual information values
        Di = np.diff(I1[i1], prepend=0.0)
        red_vars = list(inds)

        # Distribute mutual information to redundancy and synergy terms
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_s.squeeze()[t]

            if i_ == Nvars + np.count_nonzero(Di == 0) - 2 and len(ll) == 1:
                noi_ = tuple(set(inds) - set(ll))
                noj_ = tuple(set(inds) - set(lab[i_ - 1]))

                # Index i is the variable over which we are calculating sp causality
                p_i_ = p.sum(axis=(0, *noi_), keepdims=True)
                p_ti_ = p.sum(axis=noi_, keepdims=True)
                p_i__t = it.safe_div(p_ti_, p_s)
                p_t_i_ = it.safe_div(p_ti_, p_i_)

                # Index j is the variable with the largest sp mutual info after variable i
                p_j_ = p.sum(axis=(0, *noj_), keepdims=True)
                p_tj_ = p.sum(axis=noj_, keepdims=True)
                p_j__t = it.safe_div(p_tj_, p_s)
                p_t_j_ = it.safe_div(p_tj_, p_j_)

                # Rd_states is a per-pair structure (Nt, Nbins_i, Nbins_j); the
                # trigger condition can fire with ``red_vars`` of other sizes
                # when there are duplicate or zero entries in the sorted
                # specific MI, so only assign when the key is one of the
                # initialized pairs.
                rd_key = tuple(red_vars)
                if rd_key in Rd_states:
                    if Nvars != 2:
                        Rd_states[rd_key][t, :, :] = (
                            (p * it.mylog(it.safe_div(p_t_i_, p_t_j_)))
                            .sum(axis=tuple(set(inds) - set(red_vars)))
                            .squeeze()[t]
                        )
                    elif Nvars == 2:
                        p_t_j_ = p.sum(axis=(1, 2), keepdims=True)
                        Rd_states[rd_key][t, :, :] = (
                            p * it.mylog(it.safe_div(p_t_i_, p_t_j_))
                        )[t]

            if i_ == Nvars + np.count_nonzero(Di == 0) - 1 and len(ll) == 1:
                noi_ = tuple(set(inds) - set(ll))
                noj_ = tuple(set(inds) - set(lab[i_ - 1]))
                noij_ = tuple(set(inds) - set(ll) - set(lab[i_ - 1]))

                # Index i is the variable over which we are calculating sp causality
                p_i_ = p.sum(axis=(0, *noi_), keepdims=True)
                p_ti_ = p.sum(axis=noi_, keepdims=True)
                p_i__t = it.safe_div(p_ti_, p_s)
                p_t_i_ = it.safe_div(p_ti_, p_i_)

                # Index j is the variable with the largest sp mutual info after variable i
                p_j_ = p.sum(axis=(0, *noj_), keepdims=True)
                p_tj_ = p.sum(axis=noj_, keepdims=True)
                p_j__t = it.safe_div(p_tj_, p_s)
                p_t_j_ = it.safe_div(p_tj_, p_j_)

                # Joint probability of target, and sources i and j
                p_tij_ = p.sum(axis=noij_, keepdims=True)

                if Di[i_ - 1] > 1e-10:
                    Un_states[ll][t, :] = (
                        (p_tij_ * it.mylog(it.safe_div(p_t_i_, p_t_j_)))
                        .sum(axis=tuple(set(lab[i_ - 1])))
                        .squeeze()[t]
                    )
                else:
                    Un_states[ll][t, :] = (
                        (p_tij_ * it.mylog(it.safe_div(p_t_i_, p_s)))
                        .sum(axis=tuple(set(lab[i_ - 1])))
                        .squeeze()[t]
                    )

            elif i_ > Nvars + np.count_nonzero(Di == 0) - 1 and len(ll) == 2:
                noi_ = tuple(set(inds) - set(ll))
                noj_ = tuple(set(inds) - set(lab[i_ - 1]))
                noij_ = tuple(set(inds) - set(ll) - set(lab[i_ - 1]))

                # Index i is the variable over which we are calculating sp causality
                p_i_ = p.sum(axis=(0, *noi_), keepdims=True)
                p_ti_ = p.sum(axis=noi_, keepdims=True)
                p_i__t = it.safe_div(p_ti_, p_s)
                p_t_i_ = it.safe_div(p_ti_, p_i_)

                # Index j is the variable with the largest sp mutual info after variable i
                p_j_ = p.sum(axis=(0, *noj_), keepdims=True)
                p_tj_ = p.sum(axis=noj_, keepdims=True)
                p_j__t = it.safe_div(p_tj_, p_s)
                p_t_j_ = it.safe_div(p_tj_, p_j_)

                # Joint probability of target, and sources i and j
                p_tij_ = p.sum(axis=noij_, keepdims=True)

                Sy_states[ll][t, :, :] = (
                    (p_tij_ * it.mylog(it.safe_div(p_t_i_, p_t_j_)))
                    .sum(axis=tuple(set(lab[i_ - 1]) - set(ll)))
                    .squeeze()[t]
                )

            if len(ll) == 1:
                I_R[tuple(red_vars)] += info
                I_R_sp[tuple(red_vars)][t] = info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info
                I_S_sp[ll][t] = info

    return I_R, I_S, MI, info_leak, Rd_states, Un_states, Sy_states


def plot(I_R, I_S, info_leak, axs, nvars, threshold=0):
    """
    Compute and plot information flux for given data.

    Parameters
    ----------
    I_R : dict
        Data for redundant contribution.
    I_S : dict
        Data for synergistic contribution.
    info_leak : float
        Information leak value.
    axs : np.ndarray
        Axes array for plotting (shape: (nvars, 2)).
    nvars : int
        Number of variables.
    threshold : float, optional
        Threshold as a percentage of the maximum value to select
        contributions to plot (default is 0).

    Returns
    -------
    dict
        Dictionary mapping label keys to normalized values.
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            I_R_keys.append(prefix + "".join(map(str, comb)))
            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")

    # Synergestic Contributions
    I_S_keys = [
        "S" + "".join(map(str, comb))
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]
    I_S_labels = [
        f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$"
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [
        I_R.get(tuple(map(int, key[1:])), 0)
        if "U" in key or "R" in key
        else I_S.get(tuple(map(int, key[1:])), 0)
        for key in label_keys
    ]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    axs[0].set_box_aspect(1 / 4)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def nice_print(r_, s_, mi_, leak_):
    """
    Print the normalized redundancies, unique and synergy particles.

    Parameters
    ----------
    r_ : dict
        Redundancy and unique information values.
    s_ : dict
        Synergy values.
    mi_ : dict
        Mutual information values.
    leak_ : float
        Information leak value.

    Returns
    -------
    None
    """
    r_ = {key: value / max(mi_.values()) for key, value in r_.items()}
    s_ = {key: value / max(mi_.values()) for key, value in s_.items()}

    print("    Redundant (R):")
    for k_, v_ in r_.items():
        if len(k_) > 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Unique (U):")
    for k_, v_ in r_.items():
        if len(k_) == 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Synergystic (S):")
    for k_, v_ in s_.items():
        print(f"        {str(k_):12s}: {v_:5.4f}")

    print(f"    Information Leak: {leak_ * 100:5.2f}%")
    print("\n")


def run(X, nvars, nlag, nbins, axs):
    """
    Run SURD analysis on multivariate signal data.

    Parameters
    ----------
    X : np.ndarray
        Input signal data with shape (nvars, n_samples).
    nvars : int
        Number of variables.
    nlag : int
        Time lag for constructing the target variable.
    nbins : int
        Number of bins for histogram computation.
    axs : np.ndarray
        Matplotlib axes array for plotting.

    Returns
    -------
    I_R : dict
        Redundancies and unique information.
    I_S : dict
        Synergies.
    MI : dict
        Mutual information.
    info_leak : float
        Information leak.
    Rd_states : dict
        Redundancy states.
    Un_states : dict
        Unique information states.
    Sy_states : dict
        Synergy states.
    """
    information_flux = {}

    for i in range(nvars):
        print(f"SURD CAUSALITY FOR SIGNAL {i + 1}")

        # Organize data (0 target variable, 1: agent variables)
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])

        # Run SURD
        hist, _ = np.histogramdd(Y.T, nbins)
        I_R, I_S, MI, info_leak, Rd_states, Un_states, Sy_states = surd_states(hist)

        # Print results
        nice_print(I_R, I_S, MI, info_leak)

        # Plot SURD
        information_flux[i + 1] = plot(
            I_R, I_S, info_leak, axs[i, :], nvars, threshold=-0.01
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i + 1}}} / I \\left(Q_{i + 1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=12,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i + 1}}}}}{{H \\left(Q_{i + 1} \\right)}}$",
            pad=20,
        )
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=20,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )
        print("\n")

    # Show the results
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels("")

    return I_R, I_S, MI, info_leak, Rd_states, Un_states, Sy_states


def plot_states(
    data,
    bins,
    target,
    source,
    save_path,
    title,
    vmax,
    vmin,
    xlabel,
    ylabel,
    cmap,
    norm=True,
    fs=20,
):
    """
    Plot a heatmap with marginal histograms on top and right.

    Parameters
    ----------
    data : np.ndarray
        2D array of normalized values to display as heatmap.
    bins : list of np.ndarray
        Bin edges for both heatmap and marginal histograms.
    target : int
        Index of the target variable (1-based).
    source : int
        Index of the source variable (1-based).
    save_path : str
        Path where the figure will be saved.
    title : str
        Title for the top histogram.
    vmax : float
        Maximum value for heatmap color scale.
    vmin : float
        Minimum value for heatmap color scale.
    xlabel : str
        Label for the x-axis of the heatmap.
    ylabel : str
        Label for the y-axis of the heatmap.
    cmap : str or Colormap
        Colormap for the heatmap.
    norm : bool, optional
        Whether to normalize using a diverging scale centered at zero
        (default is True).
    fs : int, optional
        Font size for labels and titles (default is 20).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax_main : matplotlib.axes.Axes
        Main heatmap axes.
    ax_top : matplotlib.axes.Axes
        Top histogram axes.
    ax_right : matplotlib.axes.Axes
        Right histogram axes.
    """
    # Sum over axes for marginal histograms
    data_sum_x = data.sum(axis=0)
    data_sum_y = data.sum(axis=1)
    data = np.clip(data, vmin, vmax)  # Clip to color scale bounds

    # === Create figure and layout ===
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )

    # === Main heatmap ===
    ax_main = fig.add_subplot(gs[1, 0])
    extent = (
        bins[target - 1][0],
        bins[target - 1][-1],
        bins[source - 1][0],
        bins[source - 1][-1],
    )

    if norm:
        mynorm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        im = ax_main.imshow(
            data,
            norm=mynorm,
            cmap=cmap,
            interpolation="bicubic",
            extent=extent,
            origin="lower",
        )
    else:
        im = ax_main.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="bicubic",
            extent=extent,
            origin="lower",
        )

    ax_main.set_ylim([bins[source - 1][0], bins[source - 1][-1]])
    ax_main.set_xlim([bins[target - 1][0], bins[target - 1][-1]])
    ax_main.set_xlabel(xlabel, fontsize=fs, labelpad=0)
    ax_main.set_ylabel(ylabel, fontsize=fs, labelpad=5)
    ax_main.axvline(0, color="k", linewidth=1.5)
    ax_main.axhline(0, color="k", linewidth=1.5)

    # === Top histogram ===
    pos = ax_main.get_position()
    ax_top = fig.add_axes([pos.x0, pos.y0 + pos.height, pos.width, 0.15])
    bin_centers_x = (bins[target - 1][1:] + bins[target - 1][:-1]) / 2
    interp_x = np.linspace(bin_centers_x.min(), bin_centers_x.max(), 1000)
    interp_func_x = interp1d(
        bin_centers_x, data_sum_x, kind="cubic", fill_value="extrapolate"
    )
    interp_y = interp_func_x(interp_x)

    ax_top.plot(interp_x, interp_y, color="black", linewidth=1.25)
    ax_top.fill_between(interp_x, interp_y, 0, color=my_colors["gray"])
    ax_top.set_xlim([bin_centers_x[0], bin_centers_x[-1]])
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.axhline(0, color="k", linewidth=1.5)
    ax_top.set_title(title, fontsize=fs, pad=15)

    # === Right histogram ===
    ax_right = fig.add_axes([pos.x0 + pos.width, pos.y0, 0.15, pos.height])
    bin_centers_y = (bins[source - 1][1:] + bins[source - 1][:-1]) / 2
    interp_ygrid = np.linspace(bin_centers_y.min(), bin_centers_y.max(), 1000)
    interp_func_y = interp1d(
        bin_centers_y, data_sum_y, kind="cubic", fill_value="extrapolate"
    )
    interp_vals = interp_func_y(interp_ygrid)

    ax_right.plot(interp_vals, interp_ygrid, color="black", linewidth=1.25)
    ax_right.fill_betweenx(interp_ygrid, 0, interp_vals, color=my_colors["gray"])
    ax_right.set_xlim(left=0)
    ax_right.set_ylim([bin_centers_y[0], bin_centers_y[-1]])
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.axvline(0, color="k", linewidth=1.5)

    # === Style ===
    for ax in [ax_main, ax_top, ax_right]:
        ax.tick_params(width=1.5)
        for side in ax.spines.values():
            side.set_linewidth(1.5)

    # === Save and return ===
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return fig, ax_main, ax_top, ax_right


def plot_states_3d(data, bins, title, level=0.5, color=my_colors["redundant"]):
    """
    Plot a 3D isosurface visualization of the data.

    Parameters
    ----------
    data : np.ndarray
        3D array of values for isosurface computation.
    bins : list of np.ndarray
        Bin edges for each dimension (x, y, z).
    title : str
        Title for the plot.
    level : float, optional
        Isosurface level for marching cubes algorithm (default is 0.5).
    color : tuple
        RGB color tuple for the isosurface (default is redundant color).

    Returns
    -------
    None
    """
    # scikit-image is only needed here (optional extra ``surd[states]``).
    from skimage import measure

    x_vals, y_vals, z_vals = bins[0], bins[1], bins[2]
    dx = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    dy = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
    dz = (z_vals[-1] - z_vals[0]) / (len(z_vals) - 1)

    # Compute the isosurface
    verts, faces, normals, values = measure.marching_cubes(data, level=level)

    # Map voxel coordinates to real-world coordinates
    verts[:, 0] = x_vals[0] + verts[:, 0] * dx
    verts[:, 1] = y_vals[0] + verts[:, 1] * dy
    verts[:, 2] = z_vals[0] + verts[:, 2] * dz

    # Compute projections (maximum intensity) for each axis
    proj_x = np.max(data, axis=0)  # YZ plane
    proj_y = np.max(data, axis=1)  # XZ plane
    proj_z = np.max(data, axis=2)  # XY plane

    # Create the plot
    fig = plt.figure(figsize=(8, 5))

    # 3D isosurface view
    ax1 = fig.add_subplot(projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor(color)
    ax1.add_collection3d(mesh)
    ax1.set_xlim(x_vals[0], x_vals[-1])
    ax1.set_ylim(y_vals[0], y_vals[-1])
    ax1.set_zlim(z_vals[0], z_vals[-1])
    ax1.set_xlabel(f"$q_2^+$ $\\rm{{(target)}}$", labelpad=10)
    ax1.set_ylabel(f"$q_1$ $\\rm{{(source)}}$", labelpad=10)
    ax1.set_zlabel(f"$q_2$ $\\rm{{(source)}}$", labelpad=10)
    ax1.set_title(title)
    # 3D view
    ax1.view_init(elev=30, azim=135)

    # Draw the right-most vertical edge of the bounding box (at xmin, ymin
    # for the default ``view_init(elev=30, azim=135)``). Matplotlib's pane
    # rendering can leave this seam missing depending on version.
    ax1.plot(
        [x_vals[0], x_vals[0]],
        [y_vals[0]*1.1, y_vals[0]*1.1],
        [z_vals[0]*1.1, z_vals[-1]],
        color="black",
        linewidth=1.5,
        zorder=10,
    )

    # x_line = bins[0]
    # y_line = np.zeros_like(x_line)
    # z_line = np.zeros_like(x_line)
    # ax1.plot(x_line, y_line, z_line, color='black', linewidth=2)

    # # Meshgrid over x and z
    # X, Z = np.meshgrid(bins[0], bins[2])  # X and Z span
    # Y = np.zeros_like(X)  # Y is fixed to 0
    # ax1.plot_surface(X, Y, Z, alpha=1, color='gray', edgecolor='none', label='y=0 plane')

    # X-view
    # ax1.view_init(elev=0, azim=0)

    # Y-view
    # ax1.view_init(elev=0, azim=90)

    # Z-view
    # ax1.view_init(elev=90, azim=-90)
    ax1.set_proj_type("ortho")
    ax1.set_box_aspect([1, 1, 1], zoom=0.85)

    # Set background color and box line properties
    ax1.set_axis_on()

    # Update pane colors to white
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Set grid color and line width
    ax1.xaxis._axinfo["grid"].update(color="none")
    ax1.yaxis._axinfo["grid"].update(color="none")
    ax1.zaxis._axinfo["grid"].update(color="none")
    ax1.zaxis._axinfo["juggled"] = (1, 2, 2)

    ax1.xaxis.pane.set_edgecolor("black")
    ax1.yaxis.pane.set_edgecolor("black")
    ax1.zaxis.pane.set_edgecolor("black")
    ax1.xaxis.pane.set_alpha(1)
    ax1.yaxis.pane.set_alpha(1)
    ax1.zaxis.pane.set_alpha(1)
    ax1.xaxis.pane.set_linewidth(1.5)
    ax1.yaxis.pane.set_linewidth(1.5)
    ax1.zaxis.pane.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()
