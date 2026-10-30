import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import os, sys

    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.colors as mcolors  # type: ignore
    from surd import states
    from surd.states import analytic_eqs as cases
    from matplotlib.colors import LinearSegmentedColormap

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
        cases,
        my_colors,
        my_gray,
        np,
        os,
        plt,
        states,
    )


@app.cell
def _():
    Nt = 5 * 10**7  # Number of time steps to perform the integration of the system
    samples = Nt - 10000  # Number of samples to be considered (remove the transients)
    nbins = 51  # Number of bins to disctrize the histogram
    nlag = 1  # Time lag to perform the causal analysis
    return Nt, nbins, nlag, samples


@app.cell
def _(Nt, cases, np, os, samples):
    # Define paths for saving/loading data for each system
    formatted_Nt = "{:.0e}".format(Nt).replace("+0", "").replace("+", "")
    filepath = os.path.join("data", f"benchmark_target_Nt_{formatted_Nt}.npy")

    # Check if data is saved and load it, otherwise generate and save
    if os.path.isfile(filepath):
        X = np.load(filepath)
        print(f"Loaded data for benchmark target")
    else:
        qs = cases.target(Nt)
        X = np.array([q[-samples:] for q in qs])
        os.makedirs("data", exist_ok=True)
        np.save(filepath, X)
        print(f"Generated and saved data for benchmark target")

    nvars = X.shape[0]
    return X, nvars


@app.cell
def _(X, nbins, np):
    max_abs = np.percentile(X, 99.99)
    max_abs = np.floor(max_abs)
    bin_width = 2 * max_abs / (nbins - 1)
    bins_list = []
    for _i in range(X.shape[0]):
        bins = np.linspace(-max_abs, max_abs + bin_width, nbins + 1)
        bins_list.append(bins)
    return (bins_list,)


@app.cell
def _(X, bins_list, nlag, np, nvars, plt, states):
    Rd_results, Sy_results, mi_results, info_leak_results = ({}, {}, {}, {})
    rd_states_results, u_states_results, sy_states_results = ({}, {}, {})
    _fig, axs = plt.subplots(
        nvars,
        2,
        figsize=(10, 2.6 * nvars),
        gridspec_kw={"width_ratios": [nvars * 20, 1]},
    )
    for _i in range(nvars):
        print(f"INFORMATION FLUX FOR SIGNAL {_i + 1}")
        Y = np.vstack([X[_i, nlag:], X[:, :-nlag]])
        hist, bins_1 = np.histogramdd(
            Y.T, bins=[bins_list[_i], bins_list[0], bins_list[1], bins_list[2]]
        )
        Rd, Sy, mi, info_leak, rd_states, u_states, sy_states = states.surd_states(hist)
        states.nice_print(Rd, Sy, mi, info_leak)
        _ = states.plot(Rd, Sy, info_leak, axs[_i, :], nvars, threshold=-0.01)
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
        axs[_i, 0].set_yticks([0, 0.5])
        axs[_i, 0].set_ylim([0, 0.5])
        Rd_results[_i + 1] = Rd
        Sy_results[_i + 1] = Sy
        mi_results[_i + 1] = mi
        info_leak_results[_i + 1] = info_leak
        rd_states_results[_i + 1] = rd_states
        u_states_results[_i + 1] = u_states
        sy_states_results[_i + 1] = sy_states
    plt.tight_layout(w_pad=-12, h_pad=1)
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
        save_path=os.path.join("figures", "benchmark_target_R12_states.pdf"),
        title="$\\rm{Redundant}$  $\\rm{causality}$",
        xlabel="$q_{1}$ \\rm{(source)}",
        ylabel="$q_{2}^+$ \\rm{(target)}",
        vmax=np.max(_data),
        vmin=-np.max(_data),
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
        save_path=os.path.join("figures", f"benchmark_target_U{_source}_states.pdf"),
        title="$\\rm{Unique}$  $\\rm{causality}$",
        xlabel="$q_{1}$ \\rm{(source)}",
        ylabel="$q_{2}^+$ \\rm{(target)}",
        vmax=np.max(_data),
        vmin=-np.max(_data),
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
    _source = 2
    _data = u_states_results[_target][_source,] / Rd_results[_target][_source,]
    _data = np.maximum(_data, 0)
    _colors = [my_colors["redundant"], "#F5F4F4", my_colors["unique"]]
    _custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", _colors, N=256)
    _fig, _ax_main, _, _ = states.plot_states(
        data=_data,
        bins=bins_1,
        target=_target,
        source=_source,
        save_path=os.path.join("figures", f"benchmark_target_U{_source}_states.pdf"),
        title="$\\rm{Unique}$  $\\rm{causality}$",
        xlabel="$q_{1}$ \\rm{(source)}",
        ylabel="$q_{2}^+$ \\rm{(target)}",
        vmax=np.max(_data),
        vmin=-np.max(_data),
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
    _source = 2
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
        save_path=os.path.join("figures", f"benchmark_target_S12_states.pdf"),
        title="$\\rm{Synergistic}$  $\\rm{causality}$",
        xlabel="$q_{1}$ \\rm{(source)}",
        ylabel="$q_{2}^+$ \\rm{(target)}",
        vmax=np.max(_data),
        vmin=-np.max(_data),
        cmap=_custom_cmap,
        norm=False,
    )
    return


@app.cell
def _(Rd_results, bins_1, my_colors, np, rd_states_results, states):
    _target = 2
    _data = rd_states_results[_target][1, 2] / Rd_results[2][1, 2]
    _data = np.maximum(_data, 0)
    _level = 0.25
    _title = f"$\\Delta \\mathcal{{C}}^R_{{12\\to 2}}/\\Delta I^R_{{12\\to 2}}=$ {_level:.2f}"
    states.plot_states_3d(
        _data,
        bins_1,
        _title,
        level=_level * np.max(_data),
        color=my_colors["redundant"],
    )
    return


@app.cell
def _(Sy_results, bins_1, my_colors, np, states, sy_states_results):
    _target = 2
    _data = sy_states_results[_target][1, 2] / Sy_results[2][1, 2]
    _data = np.maximum(_data, 0)
    _level = 0.5
    _title = f"$\\Delta \\mathcal{{C}}^S_{{12\\to 2}}/\\Delta I^S_{{12\\to 2}}=$ {_level:.2f}"
    states.plot_states_3d(
        _data,
        bins_1,
        _title,
        level=_level * np.max(_data),
        color=my_colors["synergistic"],
    )
    return


if __name__ == "__main__":
    app.run()
