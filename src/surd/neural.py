"""Neural estimate of the specific mutual information for SURD.

:func:`surd.surd` estimates the specific mutual information i(t; S_A)
from the joint histogram of the target and the sources. The histogram has
``nbins**(nvars + 1)`` cells, and it needs many more samples than cells.
This module replaces the histogram by a classifier q(t | s_A) that predicts
the binned target from the continuous sources. The redundant, unique and
synergistic terms are then assigned by the same rule, :func:`surd.assign`.

Use it when the histogram is undersampled: short records, or more than
three sources. When the histogram is well sampled, it is faster and more
accurate.

Method:

- The target is binned as in :func:`surd.surd` (equal-width bins over its
  range). The sources enter as numbers, standardised.
- One network serves all subsets of sources. Its input is the source
  vector with the excluded sources set to zero, followed by the 0/1 mask of
  the subset. It is trained with cross-entropy, with a random subset drawn
  for every sample.
- Every sample is scored by a network that did not see it in training
  (contiguous-block k-fold). The specific mutual information is

      i(t; S_A) = < log2 q(t | s_A) - log2 p(t) >,

  averaged over the held-out samples with target state t. Its average over
  t, weighted by p(t), is a lower bound on I(T; S_A).
- SURD requires i(t; S_A) to grow with A. Each value is replaced by the
  maximum over the subsets of A, the empty subset included, which also
  clips it at zero.
- The leak is 1 - I(T; S)/H(T) with the estimated I(T; S), an upper bound
  on H(T | S)/H(T).

Requires PyTorch (``pip install "surd[neural]"``).

Typical use::

    import surd.neural

    # Y[0] target, Y[1:] sources, shape (nvars + 1, N)
    I_R, I_S, MI, info_leak = surd.neural.surd(Y, nbins=50)
    surd.nice_print(I_R, I_S, MI, info_leak)
"""

from itertools import combinations as icmb
from typing import Dict, Tuple

import numpy as np

from .core import assign, nice_print, plot

__all__ = ["specific_info", "surd", "run", "null_mi"]

DEFAULTS = dict(
    hidden=64,
    depth=2,
    lr=3e-3,
    weight_decay=1e-5,
    batch_size=None,
    max_steps=20000,
    max_epochs=300,
    patience=15,
    val_fraction=0.15,
    kfold=None,
    eps=1e-3,
)


def _torch():
    try:
        import torch
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "surd.neural needs PyTorch. Install it with "
            "'pip install \"surd[neural]\"' or see https://pytorch.org."
        ) from err
    return torch


def _combinations(nvars):
    inds = range(1, nvars + 1)
    return [c for n in inds for c in icmb(inds, n)]


def _bin_target(t, nbins):
    """Bin index of each target sample, as ``np.histogramdd`` would bin it."""
    if np.ndim(nbins) == 0:
        lo, hi = t.min(), t.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, int(nbins) + 1)
    else:
        edges = np.asarray(nbins, dtype=float)
    K = len(edges) - 1
    # Right-open bins with the last edge included, as in np.histogramdd.
    k = np.searchsorted(edges, t, side="right") - 1
    k[t == edges[-1]] = K - 1
    if np.any((k < 0) | (k >= K)):
        raise ValueError("target values fall outside the bin edges")
    return k, K


def _monotone(Is):
    """i(t; S_A) <- max over the subsets B of A, the empty set included."""
    out = {}
    for A in sorted(Is, key=len):
        v = np.maximum(Is[A], 0.0)
        for B in out:
            if len(B) < len(A) and set(B) <= set(A):
                v = np.maximum(v, out[B])
        out[A] = v
    return {A: out[A] for A in Is}


def _make_net(torch, nvars, K, hidden, depth):
    layers, d = [], 2 * nvars
    for _ in range(depth):
        layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
        d = hidden
    layers.append(torch.nn.Linear(d, K))
    return torch.nn.Sequential(*layers)


def _train(torch, net, X, t, masks, Xv, tv, opts, gen):
    """Cross-entropy with one random subset per sample. Early stopping on
    the validation loss averaged over all subsets."""
    F = torch.nn.functional
    n = X.shape[0]
    bs = opts["batch_size"] or int(np.clip(n // 20, 32, 1024))
    steps_per_epoch = int(np.ceil(n / bs))
    max_steps = int(min(opts["max_steps"], opts["max_epochs"] * steps_per_epoch))
    eval_every = max(50, steps_per_epoch)
    opt = torch.optim.Adam(net.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=0.02 * opts["lr"])

    def forward(x, m):
        return net(torch.cat([x * m, m], dim=1))

    best, best_state, bad, step = np.inf, None, 0, 0
    while step < max_steps:
        perm = torch.randperm(n, generator=gen, device="cpu").to(X.device)
        for b in range(steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            mi = torch.randint(len(masks), (len(idx),), generator=gen, device="cpu").to(X.device)
            loss = F.cross_entropy(forward(X[idx], masks[mi]), t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                net.eval()
                with torch.no_grad():
                    vl = np.mean([
                        F.cross_entropy(forward(Xv, m.expand_as(Xv)), tv).item() for m in masks
                    ])
                net.train()
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                if bad >= opts["patience"]:
                    step = max_steps
            if step >= max_steps:
                break
    net.load_state_dict(best_state)
    net.eval()


def specific_info(
    Y: np.ndarray,
    nbins,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
    **options,
) -> Tuple[Dict, np.ndarray, float]:
    """
    Specific mutual information between the binned target and every subset
    of the sources, estimated with a classifier.

    Parameters:
    - Y (np.ndarray): Samples, shape (nvars + 1, N). Y[0] is the target and
      Y[1:] are the sources, as in the histogram ``np.histogramdd(Y.T, ...)``.
      The samples are taken in order to build the folds, so keep them in
      time order.
    - nbins (int or array): Number of equal-width target bins, or the bin
      edges. The sources are not binned.
    - seed (int): Seed of the network initialisation and the training order.
    - device (str): PyTorch device, for instance ``"cpu"`` or ``"cuda"``.
    - verbose (bool): Print the progress over the folds.
    - options: Training options, see ``surd.neural.DEFAULTS``. ``kfold`` is
      the number of folds (default 5 for N <= 30000 and 2 above). ``eps`` is
      the weight of a uniform mixture added to q(t | s_A), a floor against
      overconfident predictions of rare states.

    Returns:
    - Is (dict): i(t; S_A) in bits, one array over the target bins per subset
      A, with the same keys as in :func:`surd.surd`.
    - p_t (np.ndarray): Probability of each target bin.
    - info_leak (float): Estimate of H(T | S)/H(T).
    """
    torch = _torch()
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown options: {sorted(unknown)}")
    opts = {**DEFAULTS, **options}

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 2:
        raise ValueError("Y must have shape (nvars + 1, N), target first")
    nvars, N = Y.shape[0] - 1, Y.shape[1]
    t_all, K = _bin_target(Y[0], nbins)
    X_all = Y[1:].T.astype(np.float32)

    combs = _combinations(nvars)
    masks_np = np.zeros((len(combs), nvars), dtype=np.float32)
    for i, c in enumerate(combs):
        masks_np[i, [j - 1 for j in c]] = 1.0

    kfold = opts["kfold"] or (5 if N <= 30000 else 2)
    folds = np.array_split(np.arange(N), kfold)
    logq = np.zeros((len(combs), N))
    eps = opts["eps"]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device(device)
    masks = torch.as_tensor(masks_np, device=dev)
    tt = torch.as_tensor(t_all, dtype=torch.long, device=dev)

    for f, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(kfold) if j != f])
        mu, sd = X_all[train].mean(0), X_all[train].std(0) + 1e-8
        X = torch.as_tensor((X_all - mu) / sd, device=dev)
        # Validation split: the last part of the training samples.
        n_val = max(8, int(opts["val_fraction"] * len(train)))
        val, fit = train[-n_val:], train[:-n_val]

        net = _make_net(torch, nvars, K, opts["hidden"], opts["depth"]).to(dev)
        _train(torch, net, X[fit], tt[fit], masks, X[val], tt[val], opts, gen)

        with torch.no_grad():
            Xte, tte = X[test], tt[test]
            for i in range(len(combs)):
                logits = net(torch.cat([Xte * masks[i], masks[i].expand_as(Xte)], dim=1))
                q = (1 - eps) * torch.softmax(logits, dim=1) + eps / K
                logq[i, test] = torch.log(q.gather(1, tte[:, None])).squeeze(1).cpu().numpy()
        if verbose:
            print(f"    fold {f + 1}/{kfold} done")

    counts = np.bincount(t_all, minlength=K).astype(float)
    p_t = counts / N
    logp = np.log(np.where(p_t > 0, p_t, 1.0))
    Is = {}
    for i, c in enumerate(combs):
        s = np.bincount(t_all, weights=logq[i], minlength=K)
        Is[c] = np.where(counts > 0, (s / np.maximum(counts, 1) - logp) / np.log(2), 0.0)

    H = -(p_t[p_t > 0] * np.log2(p_t[p_t > 0])).sum()
    MI_all = float((Is[combs[-1]] * p_t).sum())
    info_leak = float(np.clip(1.0 - MI_all / H, 0.0, 1.0)) if H > 0 else 1.0

    return _monotone(Is), p_t, info_leak


def surd(Y: np.ndarray, nbins, **kwargs) -> Tuple[Dict, Dict, Dict, float]:
    """
    SURD with the neural estimate of the specific mutual information.

    Takes the samples instead of their histogram. Parameters as in
    :func:`specific_info`. Returns I_R, I_S, MI and info_leak, as
    :func:`surd.surd` does.
    """
    Is, p_t, info_leak = specific_info(Y, nbins, **kwargs)
    I_R, I_S, MI = assign(Is, p_t)
    return I_R, I_S, MI, info_leak


def null_mi(Y: np.ndarray, nbins, seed: int = 0, **kwargs) -> float:
    """
    Mutual information I(T; S) estimated after shuffling the target against
    the sources. The true value is zero, so the result is the bias level of
    the estimate for this record length. Compare it with ``max(MI.values())``.
    """
    rng = np.random.default_rng(seed)
    Yn = np.array(Y, dtype=float)
    Yn[0] = Yn[0][rng.permutation(Yn.shape[1])]
    Is, p_t, _ = specific_info(Yn, nbins, seed=seed, **kwargs)
    return float(max((v * p_t).sum() for v in Is.values()))


def run(X, nvars, nlag, nbins, axs, **kwargs):
    """
    :func:`surd.run` with the neural estimator: SURD of each signal at time
    n + nlag against all signals at time n, printed and plotted. Extra
    keyword arguments go to :func:`specific_info`.
    """
    for i in range(nvars):
        print(f'SURD CAUSALITY FOR SIGNAL {i+1} (neural estimator)')
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])
        I_R, I_S, MI, info_leak = surd(Y, nbins, **kwargs)
        nice_print(I_R, I_S, MI, info_leak)
        plot(I_R, I_S, info_leak, axs[i, :], nvars, threshold=-0.01)
        axs[i, 0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$', pad=12)
        axs[i, 1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$', pad=20)
        axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), fontsize=20, rotation=60, ha='right', rotation_mode='anchor')
        print('\n')
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels('')
    return I_R, I_S, MI, info_leak
