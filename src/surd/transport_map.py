import numpy as np
import mpart as mt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
print('Kokkos is using', mt.Concurrency(), 'threads')

# Negative log likelihood objective
def obj(coeffs, tri_map,x):
    """ Evaluates the log-likelihood of the samples using the map-induced density. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Reference density
    rho1 = multivariate_normal(np.zeros(1),np.eye(1))   

    # Compute the map-induced density at each point
    map_of_x = tri_map.Evaluate(x)
    rho_of_map_of_x = rho1.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)

    # Return the negative log-likelihood of the entire dataset
    return -np.sum(rho_of_map_of_x + log_det)/num_points

def grad_obj(coeffs, tri_map, x):
    """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Evaluate the map
    map_of_x = tri_map.Evaluate(x)

    # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
    grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)

    # Get the gradient of the log determinant with respect to the map coefficients
    grad_log_det = tri_map.LogDeterminantCoeffGrad(x)

    return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points

def pullback_pdf(tri_map,rho,x):
    r = tri_map.Evaluate(x)
    log_pdf = rho.logpdf(r.T)+tri_map.LogDeterminant(x)
    return np.exp(log_pdf)

def transport_map(x, nbins):
    nvars = x.shape[0]      # Number of variables
    total_order = 4         # Total order of the polynomial expansion
    components = []         # To store each component of the map

    for var in range(1, nvars + 1):
        # Create a FixedMultiIndexSet for each variable up to the total order
        fixed_mset = mt.FixedMultiIndexSet(var, total_order)
        map_options = mt.MapOptions()
        S = mt.CreateComponent(fixed_mset, map_options)

        # Optimize the component
        optimizer_options = {'gtol': 1e-4, 'disp': True}
        res = minimize(obj, S.CoeffMap(), args=(S, x[:var, :]), jac=grad_obj, method='BFGS', options=optimizer_options)

        # Store the optimized component
        components.append(S)

    # Assemble the TriangularMap from the optimized components
    transport_map = mt.TriangularMap(components)
    
    # Set coefficients for the entire map (assuming obj and grad_obj functions can handle this structure)
    coeffs = np.concatenate([comp.CoeffMap() for comp in components])
    transport_map.SetCoeffs(coeffs)

    # Evaluation grid
    grids = []
    for i in range(x.shape[0]):
        grid_i = np.linspace(x[i, :].min(), x[i, :].max(), nbins)
        grids.append(grid_i)
    
    meshgrids = np.meshgrid(*grids, indexing='ij')
    grid_points = np.vstack([grid.flatten() for grid in meshgrids])

    # Compute joint pdf
    ref_distribution = multivariate_normal(np.zeros(nvars), np.eye(nvars))
    map_induced_pdf = pullback_pdf(transport_map, ref_distribution, grid_points)

    # Construct the shape tuple for reshaping
    reshaped = (nbins,) * nvars

    return map_induced_pdf.reshape(reshaped), meshgrids