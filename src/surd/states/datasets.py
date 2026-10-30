"""Data of the examples of ``surd.states``, downloaded on first use.

The turbulent boundary layer record of example E03 (two signals of
19,990,000 samples) is 148 MB and lives in the SURD-states repository,
not in the package. ``load_inner_outer`` fetches it once into a cache
directory and checks its SHA-256 before use.

The cache directory is, in this order, the ``cache_dir`` argument, the
``SURD_DATA`` environment variable, or ``~/.cache/surd``.
"""

import hashlib
import os
import shutil
import urllib.request

__all__ = ["load_inner_outer", "fetch", "cache_dir", "FILES"]

_BASE_URL = "https://github.com/ALD-Lab/SURD-states/raw/main/data/"

# File name -> (URL, SHA-256 of the file).
FILES = {
    "tbl_inner_outer_part1.mat": (
        _BASE_URL + "tbl_inner_outer_part1.mat",
        "07106d0da0c17226e3c1d51c6e748eca9acdf18df486fffc4d64d6ca9faec810",
    ),
    "tbl_inner_outer_part2.mat": (
        _BASE_URL + "tbl_inner_outer_part2.mat",
        "95e6d91c3638e7a11670e81440737506b72ff27976e2634130dc461fa902ec44",
    ),
}

# Samples per part of the inner-outer record.
_SAMPLES_PER_PART = 9_995_000
_CHUNK = 1 << 20


def cache_dir(path=None):
    """Directory where the data files are stored. Created if missing."""
    if path is None:
        path = os.environ.get("SURD_DATA") or os.path.join(
            os.path.expanduser("~"), ".cache", "surd"
        )
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, dest):
    """Download ``url`` to ``dest`` through a temporary file."""
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "surd"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        total = response.headers.get("Content-Length")
        size = f" ({int(total) / 1e6:.0f} MB)" if total else ""
        print(f"Downloading {os.path.basename(dest)}{size} to {os.path.dirname(dest)}")
        shutil.copyfileobj(response, out, _CHUNK)
    os.replace(tmp, dest)


def fetch(name, cache_dir_=None):
    """Return the local path of a data file, downloading it if needed.

    Parameters
    ----------
    name : str
        One of the keys of ``FILES``.
    cache_dir_ : str, optional
        Directory for the download. See the module docstring for the
        default.

    Returns
    -------
    str
        Path of the file. Its SHA-256 matches the one in ``FILES``.

    Raises
    ------
    OSError
        If the file cannot be downloaded, or if its checksum is wrong
        after a fresh download.
    """
    url, checksum = FILES[name]
    path = os.path.join(cache_dir(cache_dir_), name)
    if os.path.isfile(path) and _sha256(path) == checksum:
        return path
    if os.path.isfile(path):
        print(f"Checksum of {path} does not match, downloading it again")
    _download(url, path)
    actual = _sha256(path)
    if actual != checksum:
        os.remove(path)
        raise OSError(
            f"Checksum mismatch for {name}: expected {checksum}, got {actual}. "
            "The download is incomplete or the file has changed."
        )
    return path


def load_inner_outer(n_samples=None, cache_dir=None):
    """Streamwise velocity in the inner and outer layers of a boundary layer.

    Hot-wire record used in example E03 and in the Communications Physics
    paper (Fig. 5). Signal 0 is the outer-layer velocity and signal 1 the
    inner-layer velocity, both as fluctuations.

    Parameters
    ----------
    n_samples : int, optional
        Samples per signal, from the start of the record. ``None`` loads
        the full record of 19,990,000 samples. The record is split in two
        files of 9,995,000 samples, and the second one is only downloaded
        when it is needed.
    cache_dir : str, optional
        Directory for the download. See the module docstring for the
        default.

    Returns
    -------
    X : np.ndarray
        Array of shape ``(2, n_samples)`` and dtype ``float32``.
    t : np.ndarray
        Time lags in wall units, as a column of shape ``(10000, 1)``.
        ``t[k][0]`` is the lag of ``k`` samples.
    """
    import numpy as np
    import scipy.io as sio

    nparts = 1 if (n_samples is not None and n_samples <= _SAMPLES_PER_PART) else 2
    names = list(FILES)[:nparts]
    parts = [sio.loadmat(fetch(name, cache_dir)) for name in names]
    X = np.hstack([p["u"] for p in parts])[:, :n_samples]
    t = parts[0]["t"]
    return X, t
