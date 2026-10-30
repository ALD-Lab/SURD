import numpy as np

np.random.seed(10)


def source(N):
    """
    Generate source signal data.

    Parameters
    ----------
    N : int
        Number of time steps.

    Returns
    -------
    q1 : np.ndarray
        First source variable.
    q2 : np.ndarray
        Second source variable.
    q3 : np.ndarray
        Third source variable.
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = (
        np.random.normal(0, 1, N),
        np.random.normal(-2, 1, N),
        np.random.normal(2, 1, N),
    )
    for n in range(N - 1):
        q1[n + 1] = W1[n]
        q2[n + 1] = (
            np.sin(1 * q1[n]) + 0.1 * W2[n] if q1[n] > 0 else 0.9 * q2[n] + 0.1 * W2[n]
        )
        q3[n + 1] = (
            np.cos(1 * q1[n]) + 0.1 * W3[n] if q1[n] < 0 else 0.9 * q3[n] + 0.1 * W3[n]
        )
    return q1, q2, q3


def target(N):
    """
    Generate target signal data.

    Parameters
    ----------
    N : int
        Number of time steps.

    Returns
    -------
    q1 : np.ndarray
        First variable.
    q2 : np.ndarray
        Second variable.
    q3 : np.ndarray
        Third variable.
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = (
        np.random.normal(0, 1, N),
        np.random.normal(-2, 1, N),
        np.random.normal(2, 1, N),
    )
    for n in range(N - 1):
        q1[n + 1] = W1[n]
        q2[n + 1] = q1[n] * np.sin(q1[n]) + 0.1 * W2[n]
        q2[n + 1] = q2[n + 1] if q2[n + 1] > 0 else q2[n] + 0.1 * W2[n]
        q3[n + 1] = q1[n] * np.cos(q1[n]) + 0.1 * W3[n]
        q3[n + 1] = q3[n + 1] if q3[n + 1] < 0 else q3[n] + 0.1 * W3[n]
    return q1, q2, q3
