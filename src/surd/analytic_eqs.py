import numpy as np
np.random.seed(10)

def mediator(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-1):
        q1[n+1] = np.sin(q2[n]) + 0.001*W1[n]
        q2[n+1] = np.cos(q3[n]) + 0.01*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def confounder(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-1):
        q1[n+1] = np.sin(q1[n] + q3[n]) + 0.01*W1[n]
        q2[n+1] = np.cos(q2[n] - q3[n]) + 0.01*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def synergistic_collider(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-1):
        q1[n+1] = np.sin(q2[n] * q3[n]) + 0.001*W1[n]
        q2[n+1] = 0.5*q2[n] + 0.1*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def redundant_collider(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-1):
        # q1[n+1] = 0.3*q1[n] + 0.7*(np.sin(q2[n]*q3[n]) + 0.1*W1[n])
        q1[n+1] = 0.3*q1[n] + (np.sin(q2[n]*q3[n]) + 0.001*W1[n])
        q2[n+1] = 0.5*q2[n] + 0.1*W2[n]
        q3[n+1] = q2[n+1]
    return q1, q2, q3