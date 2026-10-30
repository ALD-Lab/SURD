import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import os, sys, pymp

    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.colors as mcolors  # type: ignore
    from surd import states
    from surd.states import analytic_eqs as cases
    from matplotlib.colors import LinearSegmentedColormap
    from surd.states import it_tools as it

    np.random.seed(10)

    # Configure matplotlib to use LaTeX for text rendering and set font size
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 22})

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
    return (
        LinearSegmentedColormap,
        it,
        my_colors,
        my_gray,
        np,
        os,
        plt,
        pymp,
        states,
    )


@app.cell
def _(states):
    # Samples per signal. None uses the full record (19,990,000), as in the paper.
    # N_SAMPLES = 3_000_000 runs faster but shifts the outer-to-inner peak from Delta t+ = 58 to 37.
    N_SAMPLES = None

    X, time_plus = states.datasets.load_inner_outer(N_SAMPLES)
    nvars = X.shape[0]
    nbins = 25
    max_lag = 100
    return X, max_lag, nbins, nvars, time_plus


@app.cell
def _(X, nbins, np):
    max_abs = np.percentile(X, 99.9)
    max_abs = np.floor(max_abs)
    bin_width = 2 * max_abs / (nbins - 1)  # Calculate the bin width
    bins_list = []
    for _i in range(X.shape[0]):
        if _i == 0:
            lower_limit = np.percentile(X[_i, :], 0.01)
            upper_limit = np.percentile(X[_i, :], 99.99)
        else:
            lower_limit = np.percentile(X[_i, :], 0.001)
            upper_limit = np.percentile(X[_i, :], 98.5)
        bin_width = (upper_limit - lower_limit) / (nbins - 1)
        bins = np.linspace(lower_limit, upper_limit + bin_width, nbins + 1)
        bins_list.append(bins)
    return (bins_list,)


@app.cell
def _(
    X,
    bins_list,
    it,
    max_lag,
    my_colors,
    np,
    nvars,
    plt,
    pymp,
    states,
    time_plus,
):
    # Select delta T
    nlags_range = range(1, max_lag, 1)
    num_lags = len(nlags_range)
    unique_lag = pymp.shared.array((nvars, num_lags), dtype=np.float64)
    # Initialize a shared array with dimensions [nvars, nlags]
    synergy_lag = pymp.shared.array((nvars, num_lags), dtype=np.float64)
    self_unique_lag = pymp.shared.array((nvars, num_lags), dtype=np.float64)
    with pymp.Parallel(2) as par:
        for _i in par.range(nvars):
            # === Causality analysis across lags ===
            for n_idx, nlag in enumerate(nlags_range):
                _Y = np.vstack([X[_i, nlag:], X[:, :-nlag]])
                _hist, _ = np.histogramdd(
                    _Y.T, bins=[bins_list[_i], bins_list[0], bins_list[1]]
                )
                I_R, I_S, MI, _info_leak, *_ = states.surd_states(
                    _hist
                )  # Prepare lagged joint data
                H = it.entropy_nvars(_hist / _hist.sum(), (0,))
                single_keys = [k for k in I_R if len(k) == 1 and k != (_i + 1,)]
                unique_lag[_i, n_idx] = (
                    sum((I_R[k] for k in single_keys)) / H
                )  # Compute joint histogram
    _fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        time_plus[: max_lag - 1],
        unique_lag[0, :],
        color=my_colors["redundant"],
        linewidth=3,
        label="$\\rm inner \\rightarrow outer$",
    )
    ax.plot(
        time_plus[: max_lag - 1],
        unique_lag[1, :],
        color=my_colors["unique"],
        linewidth=3,
        label="$\\rm outer \\rightarrow inner$",
    )  # Compute SURD decomposition
    ax.set_xlim(0, max_lag)
    for side in ["top", "bottom", "left", "right"]:  # Entropy of target
        ax.spines[side].set_linewidth(2)
    leg = ax.legend(
        loc="best", edgecolor="white", handlelength=1.25, fontsize=18
    )  # Unique causality from all other sources
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1, 1, 1, 1))
    plt.tight_layout()
    plt.show()
    return (unique_lag,)


@app.cell
def _(time_plus, unique_lag):
    # === Find max lag index for cross causality (outer → inner) ===
    # nlag = 39
    nlag_1 = unique_lag[1, :].argmax()
    print(
        f"Maximum cross-induced causality at lag index {nlag_1}, which is equivalent to {time_plus[nlag_1][0]:.2f} time plus units"
    )
    return (nlag_1,)


@app.cell
def _(X, bins_list, nlag_1, np, nvars, plt, states):
    Rd_results, Sy_results, mi_results, info_leak_results = ({}, {}, {}, {})
    rd_states_results, u_states_results, sy_states_results = ({}, {}, {})
    _fig, axs = plt.subplots(
        nvars, 2, figsize=(7, 2.5 * nvars), gridspec_kw={"width_ratios": [35, 1]}
    )
    for _i in range(nvars):
        print(f"INFORMATION FLUX FOR SIGNAL {_i + 1}")
        _Y = np.vstack([X[_i, nlag_1:], X[:, :-nlag_1]])
        _hist, bins_1 = np.histogramdd(
            _Y.T, bins=[bins_list[_i], bins_list[0], bins_list[1]]
        )
        Rd, Sy, mi, _info_leak, rd_states, u_states, sy_states = states.surd_states(_hist)
        states.nice_print(Rd, Sy, mi, _info_leak)
        _ = states.plot(Rd, Sy, _info_leak, axs[_i, :], nvars, threshold=-0.01)
        axs[_i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {_i + 1}}} / I \\left(Q_{_i + 1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=12,
        )
        axs[_i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {_i + 1}}}}}{{H \\left(Q_{_i + 1}^+ \\right)}}$",
            pad=20,
        )
        axs[_i, 0].set_xticklabels(
            axs[_i, 0].get_xticklabels(),
            fontsize=18,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )
        axs[_i, 0].set_yticks([0, 1])
        axs[_i, 0].set_ylim([0, 1])
        axs[_i, 0].set_box_aspect(1 / 1.75)
        for j in range(0, nvars - 1):
            axs[j, 0].set_xticklabels("")
        Rd_results[_i + 1] = Rd
        Sy_results[_i + 1] = Sy
        mi_results[_i + 1] = mi
        info_leak_results[_i + 1] = _info_leak
        rd_states_results[_i + 1] = rd_states
        u_states_results[_i + 1] = u_states
        sy_states_results[_i + 1] = sy_states
    axs[0, 0].set_title(
        f"${{\\Delta I}}_{{(\\cdot) \\rightarrow O}} / I \\left(u_O^+ ; \\mathrm{{\\mathbf{{u}}}} \\right)$",
        pad=15,
    )
    axs[0, 1].set_title(
        f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow O}}}}{{H \\left(u_O^+\\right)}}$",
        pad=20,
    )
    axs[1, 0].set_title(
        f"${{\\Delta I}}_{{(\\cdot) \\rightarrow I}} / I \\left(u_I^+ ; \\mathrm{{\\mathbf{{u}}}} \\right)$",
        pad=15,
    )
    axs[1, 1].set_title(
        f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow I}}}}{{H \\left(u_I^+\\right)}}$",
        pad=20,
    )
    axs[0, 0].set_title(
        f"${{\\rm Causality\\,\\,to\\,\\,}} u_O^+$", fontsize=20, pad=15
    )
    axs[0, 1].set_title(f"$\\rm Leak$", pad=15, fontsize=20)
    axs[1, 0].set_title(
        f"${{\\rm Causality\\,\\,to\\,\\,}} u_I^+$", fontsize=20, pad=15
    )
    axs[1, 1].set_title(f"$\\rm Leak$", pad=15, fontsize=20)
    axs[1, 0].set_xticklabels(
        [
            f"$\\rm Redundant$",
            f"${{\\rm Unique\\,\\,from\\,\\,}} u_O$",
            f"${{\\rm Unique\\,\\,from\\,\\,}} u_I$",
            f"$\\rm Synergistic$",
        ],
        fontsize=14,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    plt.tight_layout(w_pad=-8, h_pad=0.3)
    plt.show()
    return (
        Rd_results,
        Sy_results,
        bins_1,
        rd_states_results,
        sy_states_results,
        u_states_results,
    )


@app.cell
def _(
    LinearSegmentedColormap,
    Rd_results,
    bins_1,
    my_colors,
    my_gray,
    np,
    os,
    rd_states_results,
    states,
):
    _target = 2
    _source = 1
    _data = rd_states_results[_target][1, 2] / Rd_results[_target][1, 2]
    _data = np.maximum(_data, 0)
    _pos = np.argmax(_data.sum(axis=(0, 1)))
    _colors = [my_gray, "#F5F4F4", my_colors["redundant"]]
    _custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", _colors, N=256)
    _fig, _ax_main, _, _ = states.plot_states(
        data=_data[:, :, _pos],
        bins=bins_1,
        target=_target,
        source=_source,
        save_path=os.path.join("figures", "tbl_R12_states.pdf"),
        title="$\\rm{Redundant}$  $\\rm{causality}$",
        xlabel=f"${{u_O}}'$",
        ylabel=f"${{u_I^+}}'$",
        vmax=0.5 * np.max(_data),
        vmin=-0.5 * np.max(_data),
        cmap=_custom_cmap,
        norm=False,
    )
    return


@app.cell
def _(
    LinearSegmentedColormap,
    Rd_results,
    bins_1,
    my_colors,
    np,
    os,
    states,
    u_states_results,
):
    _target = 2
    _source = 1
    _data = u_states_results[_target][_source,] / Rd_results[_target][_source,]
    _data = np.maximum(_data, 0)
    _colors = [my_colors["redundant"], "#F5F4F4", my_colors["unique"]]
    _custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", _colors, N=256)
    _fig, _ax_main, _, _ = states.plot_states(
        data=_data,
        bins=bins_1,
        target=_target,
        source=_source,
        save_path=os.path.join("figures", f"tbl_U{_source}_states.pdf"),
        title="$\\rm{Unique}$  $\\rm{causality}$",
        xlabel=f"${{u_O}}'$",
        ylabel=f"${{u_I^+}}'$",
        vmax=0.5 * np.max(_data),
        vmin=-0.5 * np.max(_data),
        cmap=_custom_cmap,
        norm=False,
    )
    return


@app.cell
def _(
    LinearSegmentedColormap,
    Sy_results,
    bins_1,
    my_colors,
    np,
    os,
    states,
    sy_states_results,
):
    _target = 2
    _source = 1
    _data = sy_states_results[_target][1, 2] / Sy_results[_target][1, 2]
    _data = np.maximum(_data, 0)
    _pos = np.argmax(_data.sum(axis=(0, 1)))
    _colors = [my_colors["unique"], "#F5F4F4", my_colors["synergistic"]]
    _custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", _colors, N=256)
    _fig, _ax_main, _, _ = states.plot_states(
        data=_data[:, :, _pos],
        bins=bins_1,
        target=_target,
        source=_source,
        save_path=os.path.join("figures", "tbl_S12_states.pdf"),
        title="$\\rm{Redundant}$  $\\rm{causality}$",
        xlabel=f"${{u_O}}'$",
        ylabel=f"${{u_I^+}}'$",
        vmax=0.5 * np.max(_data),
        vmin=-0.5 * np.max(_data),
        cmap=_custom_cmap,
        norm=False,
    )
    return


if __name__ == "__main__":
    app.run()
