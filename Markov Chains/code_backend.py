import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from functools import partial
import numpy as np
from jax import grad, hessian
from numba import njit


@njit(fastmath=True)
def log_likelihood(y, x, q):
  """ Likelihood for y = {0, 1}^n
  """
  p = y * 2 - 1  #translate y to -1, 1
  return - np.log(1 + np.exp(- p * (x @ q))).sum()

@njit(fastmath=True)
def grad_log_likelihood(y, x, q):
  p = y * 2 - 1 #translate y to -1, 1
  return (x * ( p * (1 + np.exp(p * (x @ q)))** (-1) ).reshape((-1,1))).sum(axis = 0)

@njit(fastmath=True)
def hessian_log_likelihood(y, x, q):
  p = y * 2 - 1
  return  (p.reshape((-1,1))* x).T @ ( x * (p * ((1 + np.exp( - p * (x @ q)))** (-1))).reshape((-1,1)))

@njit(fastmath=True)
def precompute(y, x, qs):
  """ return val, primal, gradient and hessian at point qs
  """
  px =  (y * 2 - 1 ).reshape(-1,1) * x 
  u = 1 + np.exp(- px @ qs)
  v = u ** (-1)
  return qs, -np.log(u).sum(), (px * (1 - v).reshape(-1,1)).sum(axis=0), - (px.T * (v * (1 - v))) @ px 

@njit(fastmath=True)
def hat_diff(u, q, qs, prim, grad, hess, x, y):
  """log-likelihood difference estimator.
  """
  n = x.shape[0]
  m = u.shape[0]
  diff = log_likelihood(y[u], x[u], q) - control_v(q, *precompute(y[u], x[u], qs))
  return control_v(q, qs, prim, grad, hess) + (n / m) * diff

@njit(fastmath=True)
def grad_hat_diff(u, q, qs, prim, grad, hess, x, y):
  """ Estimator of the gradient. 
  In our case, it is also the gradient of the estimator. 
  Assumes that hessian is a symetric matrix (which should be the case.)
  """
  n = x.shape[0]
  m = u.shape[0]
  grad_control_full = grad + hess @  (q - qs)
  _, _, grad_batch, hess_batch = precompute(y[u], x[u], qs)
  grad_control_batch = grad_batch + hess_batch @ (q - qs)
  grad_diff_batch = grad_log_likelihood(y[u], x[u], q) - grad_control_batch 
  return grad_control_full + (n / m) * grad_diff_batch

@njit(fastmath=True)
def control_v(q, qs, prim, grad, hess):
  """ 
  2nd order approximation of the loglikelihood at qs, evaluated at q
  given the precomputed primal, gradient and hessian.
  """
  d = q - qs
  return prim + grad @ d + 0.5 * d.T @ (hess @ d)





  #I'm taking the log here, but maybe I should be more carefull not to introduce some
#biais
def log_block_poisson(u, lmbda, a, l, q, qs, prim, grad, hess):
  """ logarithm of block-poisson estimator.
      expects u to be a list of list of vectors of indices.(lambda * xi_lamda)
  """
  qk_sum = control_v(q, qs, prim, grad, hess)
  sum_log_diff = 0
  for ul in u:
    for ui in ul:
      diff = log_likelihood(y[ui], x[ui], q) - control_v(q, *precompute(y[ui], x[ui], qs))
      sum_log_diff += np.log((diff - a) / lmbda)

  return qk_sum + (a + lmbda) + sum_log_diff




def grad_log_block_poisson(u, lmbda, a, l, q, qs, prim, grad, hess):
  """ gradient of the log_block_poisson estimator.
  """
  n = x.shape[0]
  grad_qk_sum = grad + hess @  (q - qs)
  grad_sum_log_diff = 0.0
  for ul in u:
    for ui in ul:
      qs, v, grad_batch, hess_batch = precompute(y[u], x[u], qs)
      diff = log_likelihood(y[ui], x[ui], q) - control_v(q, qs, v, grad_batch, hess_batch)
      grad_control_batch = grad_batch + hess_batch @ (q - qs)
      grad_diff_batch = grad_log_likelihood(y[u], x[u], q) - grad_control_batch 
      grad_sum_log_diff += grad_diff_batch / (diff - a) * lmbda
  return grad_qk_sum + grad_sum_log_diff








@njit(fastmath=True)
def block_update(u, G, x):
  """block-update of u considering G blocks. 
  """
  m, n = u.shape[0], x.shape[0]
  group_size = m // G
  replace = np.arange(m) < group_size
  new_u = np.random.choice(n, u.shape, replace=False)
  replace = np.random.permutation(replace)
  return np.where(replace, new_u, u)







def poisson_update(u, m, lmbda, kappa):
  """update of u for a poisson law. 
  m is the size of the base batch.
  only kappa list of batches are updated to correlate u's.
  correlation ~ 1 - kappa / lmbda
  returns a list of list of batches. (lambda * xi_lambda batches)
  """
  n = x.shape[0]
  indices_to_update = np.random.choice(lmbda, kappa, replace=False)
  
  xi = np.random.poisson(1, kappa)
  for idx in indices_to_update:
    u[idx] = []
    for j in range(xi):
      batch = np.random.choice(n, m, replace=False)
      u[idx].append(batch)
  return u






  #to do: check if using a cholesky decomposition of M is better.
@njit(fastmath=True)
def leapfrog(q, p, u, p_est, e, D, L, shrink, x, y):

  """ 
  Leapfrog approximator, step size e, length D*e.
  q: cinetic variable.
  p: momemtum variable.
  L: cholesky lower part of M
  grad_U: gradient of the cinetic energy.
  """
  #M_i = np.linalg.inv(L.T @ L) weird, it should be L.T @ L to give me M.
  for i in range(D):
    p -= 0.5 * e * grad_U(q, p, u, p_est, shrink, x, y )
    q += e * np.linalg.solve(L.T, np.linalg.solve(L, p)) #should be doing a fast triangular solve instead. #or just get the inverse since the matrix is small
    p -= 0.5 * e * grad_U(q, p, u, p_est, shrink, x, y)
  return q, -p

@njit(fastmath=True)
def leapfrog_exact(q, p, e, D, L, x, y):
  for i in range(D):
    p -= 0.5 * e * grad_log_likelihood(y, x, q)
    q += e * np.linalg.solve(L.T, np.linalg.solve(L, p)) #should be doing a fast triangular solve instead. #or just get the inverse since the matrix is small
    p -= 0.5 * e * grad_log_likelihood(y, x, q)
  return q, -p



@njit(fastmath=True)
def U(q, p, u, p_est, shrink, x, y):
  """
  potential part of the batch-approximated Hamiltonian, assuming a gaussian prior of variance shrink**(-2)I.
  (Correct at a constant; only supposed to be used in deltas.)
  """
  neg_log_likelihood =  - hat_diff(u, q, *p_est, x, y)
  neg_log_prior = + shrink**2 * 0.5 * np.linalg.norm(q, 2) ** 2
  return neg_log_likelihood + neg_log_prior

@njit(fastmath=True)
def grad_U(q, p, u, p_est, shrink,  x, y):
  grad_like = grad_hat_diff(u, q, *p_est,  x, y)
  grad_prior = + shrink**2 * q
  return - grad_like + grad_prior

@njit(fastmath=True)
def hess_U(p_est, shrink):
  _, val, grad, hess = p_est
  return - hess + shrink**2 * np.identity(hess.shape[0])

#to do: check if using a cholesky decomposition of M is better.
@njit(fastmath=True)
def K(q, p, u, L): 
  """
  Momentum part of the batch-approximated Hamiltonian, assuming a multivariate
  gaussian of variance M
  (Correct at a constant; only supposed to be used in deltas.)
  """ 
  #Should use solve triangular but not implemented in numba...
  #computes p.T @ 
  #print((0.5 * p.T @ M_i @ p),
  return  0.5 * np.linalg.norm(np.linalg.solve(L, p), 2) ** 2
@njit
def H(q, p, u, p_est, shrink, L, x, y):
  """Full batch-approximated Hamiltonian"""
  return U(q, p, u, p_est, shrink, x, y) + K(q, p, u, L)

@njit
def H_exact(q, p, shrink, L, y, x):
  neg_log_likelihood =  - log_likelihood(y, x, q)
  neg_log_prior = + shrink**2 * 0.5 * np.linalg.norm(q, 2) ** 2
  U = neg_log_likelihood + neg_log_prior
  K = 0.5 * np.linalg.norm(np.linalg.solve(L, p), 2) ** 2
  return U + K




def run(u, q, qs, PRIM, GRAD, HESS,L , i, e, Hs, n_iter, alpha, y, x, shrink, G, lmbda, log_delta, t0, nu, gamma, kappa):
  """ Runs a single chain. Obtain the input parameters using the init func.
      Runs for 2000 iterations
      with full dataset updates every 100 iterations.
  """
  ul, ql, el, hl, lgal = (np.zeros((n_iter, u.shape[0])),
                          np.zeros((n_iter, q.shape[0])),
                          np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter))
  
  for iteri in range(n_iter):
      if iteri % alpha == 0:
          qs, PRIM, GRAD, HESS, L , i = update(y, x, q, shrink)
          p_est = (qs, PRIM, GRAD, HESS)

      u, q, e, Hs, i, lga = hmc_ec_kernel(u, q, p_est, e, L, Hs, i, G, x, y, lmbda, shrink, log_delta, t0, nu, gamma, kappa)
      ul[iteri] = u
      ql[iteri] = q
      el[iteri] = e
      hl[iteri] = Hs
      lgal[iteri] = lga
  return ul, ql, el, hl, lgal

def run_exact(q,L , i, e, Hs, n_iter, alpha, y, x, shrink, G, lmbda, log_delta, t0, nu, gamma, kappa):
  """ Runs a single chain. Obtain the input parameters using the init func.
      Runs for 2000 iterations
      with full dataset updates every 100 iterations.
  """
  ql, el, hl, lgal = (np.zeros((n_iter, q.shape[0])),
                          np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter))
  
  for iteri in range(n_iter):
      if iteri % alpha == 0:
          qs, PRIM, GRAD, HESS, L , i = update(y, x, q, shrink)

      q, e, Hs, i, lga = hmc_kernel(q, e, L, Hs, i, G, x, y, lmbda, shrink, log_delta, t0, nu, gamma, kappa)
      ql[iteri] = q
      el[iteri] = e
      hl[iteri] = Hs
      lgal[iteri] = lga
  return ql, ql, el, hl, lgal







  #freely inspired by https://rlouf.github.io/post/jax-random-walk-metropolis/
#to do: optimize calls to (hat_log(...) - hat_log(...))
@njit
def multivariate_normal(L): #sample from the multivariate gaussian: p' = Lp
  u = np.array([ np.random.normal(0, 1) for _ in range(L.shape[0])])
  return (L @ u)


def hmc_ec_kernel(u, q, p_est, e, L, Hs, i, G, x, y, lmbda, shrink, log_delta, t0, nu, gamma, kappa):
    
    #pseudo-step
    u_prime = block_update(u.copy(), G, x)
    log_uniform = np.log(np.random.uniform(0,1))
    log_alpha_pseudo = hat_diff(u_prime, q, *p_est, x, y) - hat_diff(u, q, *p_est, x, y)
    do_accept = log_uniform < log_alpha_pseudo
    if do_accept:
      u = u_prime
    
    #hamiltonian step
    p = multivariate_normal(L) #to do: optimize the inversion away.
    qL, pL = leapfrog(q.copy(), p.copy(), u, p_est, e, np.minimum(int(lmbda/e)+1, 10), L, shrink, x, y)
    log_uniform = np.log(np.random.uniform(0,1))
    log_alpha_hmc = - H(qL, pL, u, p_est, shrink, L, x, y) + H(q, p, u, p_est, shrink, L, x, y)
    do_accept = log_uniform  < log_alpha_hmc
    if do_accept:
      q = qL
    
    #dual averaging (not done for the moment, some nans errors) with some diffrence (log / exp)
    #exponential averaging of the errors
    #as to be re-initialized each time we update H.
    Hs_next = np.exp(log_delta) - np.minimum(np.exp(log_alpha_hmc), 1.0)
    Hs = (1 - (1/(i + t0))) * Hs + (1/(i + t0)) * Hs_next
    log_e = nu - (i**(0.5) / gamma) * Hs
    e_next = i**(-kappa) * log_e + (1 - i**(-kappa)) * np.log(e)
    e = np.exp(e_next)
    return u, q, e, Hs, i+1.0, log_alpha_hmc


@njit
def hmc_kernel(q, e, L, Hs, i, G, x, y, lmbda, shrink, log_delta, t0, nu, gamma, kappa):
    
    
    #hamiltonian step
    p = multivariate_normal(L) #to do: optimize the inversion away.
    qL, pL = leapfrog_exact(q.copy(), p.copy(), e, np.minimum(int(lmbda/e)+1, 10), L,  x, y)
    log_uniform = np.log(np.random.uniform(0,1))
    log_alpha_hmc = - H_exact(qL, pL, shrink, L, y, x) + H_exact(q, p, shrink, L, y, x)
    do_accept = log_uniform  < log_alpha_hmc
    if do_accept:
      q = qL
    
    #dual averaging (not done for the moment, some nans errors) with some diffrence (log / exp)
    #exponential averaging of the errors
    #as to be re-initialized each time we update H.
    Hs_next = np.exp(log_delta) - np.minimum(np.exp(log_alpha_hmc), 1.0)
    Hs = (1 - (1/(i + t0))) * Hs + (1/(i + t0)) * Hs_next
    log_e = nu - (i**(0.5) / gamma) * Hs
    e_next = i**(-kappa) * log_e + (1 - i**(-kappa)) * np.log(e)
    e = np.exp(e_next)
    return q, e, Hs, i+1.0, log_alpha_hmc



def init_exact(y, x, shrink):
  """
  return all the initial parameters necessary to start a chain.
  k: inital rng seed.
  """
  m = int(x.shape[0] ** 0.5) # choice of m for the diff estimator.
  u = np.arange(20)
  q = np.zeros(x.shape[1])
  #q = q_good.reshape((4,))
  qs = q
  qs, PRIM, GRAD, HESS = precompute(y, x, qs)
  p_est = (qs, PRIM, GRAD, HESS)
  
  #https://arxiv.org/pdf/1206.1901.pdf: 
  #set the mass matrix as the inverse of the posterior variance
  #which is close to the negative hessian of the loglikelihood, 
  #or the hessian of the potential energy.
  #our paper is very not clear about that. 
  M = hess_U(p_est, shrink)
    
  L = np.linalg.cholesky(M)
  
  e = 1e-3
  i = 1.0
  Hs = 0.0

  return q, L, i, e, Hs

def init(y, x, shrink):
  """
  return all the initial parameters necessary to start a chain.
  k: inital rng seed.
  """
  m = int(x.shape[0] ** 0.5) # choice of m for the diff estimator.
  u = np.arange(20)
  q = np.zeros(x.shape[1])
  #q = q_good.reshape((4,))
  qs = q
  qs, PRIM, GRAD, HESS = precompute(y, x, qs)
  p_est = (qs, PRIM, GRAD, HESS)
  
  #https://arxiv.org/pdf/1206.1901.pdf: 
  #set the mass matrix as the inverse of the posterior variance
  #which is close to the negative hessian of the loglikelihood, 
  #or the hessian of the potential energy.
  #our paper is very not clear about that. 
  M = hess_U(p_est, shrink)
    
  L = np.linalg.cholesky(M)
  
  e = 1e-3
  i = 1.0
  Hs = 0.0

  return u, q, qs, PRIM, GRAD, HESS, L, i, e, Hs

@njit
def update(y, x, q, shrink):
  """ Recomputes the primal, gradient and hessian of the log-likelihood over 
  the full dataset at point q. Used to update the likelihood estimator.
  Does also:
    * Recompute M, the variance of the momentum to be the hessian of U, the cinetic energy 
    * Reset the iteration counter to zero to reset the dual averaging scheme.
  """
  qs = q
  qs, PRIM, GRAD, HESS = precompute(y, x, qs)
  p_est = (qs, PRIM, GRAD, HESS)
  M = hess_U(p_est, shrink)
  L = np.linalg.cholesky(M)
  i = 1.0
  return qs, PRIM, GRAD, HESS, L, i




def plot_distribution(samples, samples_exact, f, k, q_good):
	ul, ql, el, hl, lgal = samples
	ul_e, ql_e, el_e, hl_e, lgal_e = samples_exact
	fig, axs = plt.subplots(1, 4, figsize=(32, 8))
	for i, ax in enumerate(axs):
	    ax.hist(ql[f:k, i],label="EC density", bins=20,  alpha=0.7, density = True)
	    ax.hist(ql_e[f:k, i],label="HMC density", bins=20,  alpha=0.5, color = "black", density = True)
	    ax.axvline(q_good[0,i], c='r', label="sklearn logistic coeff")
	    ax.axvline(np.median(ql[f:k,i]), c="black", label="EC median ")
	    ax.axvline(np.median(ql_e[f:k,i]), c="green", label="HMC median")
	_ = fig.suptitle("samples repartions and logistic regression coeffs")
	_ = fig.legend(*ax.get_legend_handles_labels())



def plot_parameters(samples, samples_exact, f, k):
	ul, ql, el, hl, lgal = samples
	ul_e, ql_e, el_e, hl_e, lgal_e = samples_exact
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(32, 32))
	#ECs
	el_plot = el[0::200]
	ql_plot = ql[0::200]
	lgal_plot = lgal[0::200]

	_ = ax1.semilogy(el_plot[f:k,], c='r')
	_ = ax1.twinx().plot(np.exp(lgal_plot[f:k,]),",")
	ax1.set_title("acceptance ratio & step size")
	plt.ylim(0, 1.2)
	_ = ax2.plot(ql_plot[f:k,])
	ax2.set_title("parameters")

	# Exaxct
	el_plot_e = el_e[0::200]
	ql_plot_e = ql_e[0::200]
	lgal_plot_e = lgal_e[0::200]
	_ = ax3.semilogy(el_plot_e[f:k,], c='r')
	_ = ax3.twinx().plot(np.exp(lgal_plot_e[f:k,]),",")
	ax3.set_title("acceptance ratio & step size")
	plt.ylim(0, 1.2)
	_ = ax4.plot(ql_plot_e[f:k,])
	ax4.set_title("parameters")

	plt.show()

def plot_acceptance(samples):
    ul, ql, el, hl, lgal = samples
    _ = plt.plot(np.minimum(np.exp(lgal_plot),1),".")
    _ = plt.ylim(0, 2)
    _ = plt.title(f"Acceptance ratio value, mean = {np.minimum(np.exp(lgal), 1).mean():.2f}")
    _ = plt.xlabel("iteration")
    _ = plt.ylabel("value")
    _ = plt.axhline(0.65, c='r', label="0.65")
    _ = plt.legend()

def plot_acceptance(samples):
    ul, ql, el, hl, lgal = samples
    el_plot = el[0::200]
    ql_plot = ql[0::200]
    lgal_plot = lgal[0::200]
    _ = plt.plot(np.minimum(np.exp(lgal_plot),1),".")
    _ = plt.ylim(0, 2)
    _ = plt.title(f"Acceptance ratio value, mean = {np.minimum(np.exp(lgal), 1).mean():.2f}")
    _ = plt.xlabel("iteration")
    _ = plt.ylabel("value")
    _ = plt.axhline(0.65, c='r', label="0.65")
    _ = plt.legend()


def plot_avg(samples, samples_exact):
    ul, ql, el, hl, lgal = samples
    ul_e, ql_e, el_e, hl_e, lgal_e = samples_exact
    avg_params, avg_params_exact= np.array([]), np.array([])

    for param in ql.T:
        avg_params = np.append(avg_params, np.average(param))
    for param in ql_e.T:
        avg_params_exact = np.append(avg_params_exact, np.average(param))
    _ = plt.plot(avg_params, avg_params_exact, "." )
    _ = plt.ylim(0, 2)
    _ = plt.plot([-1,1], [-1,1], "-")
    _ = plt.axis([-0.1,0.1,-.2, .2])
    _ = plt.title(f"Average parameters of EC in function of HMC")
    _ = plt.xlabel("iteration")
    _ = plt.ylabel("value")
    _ = plt.legend()




def plot_variance(samples, samples_exact):
    ul, ql, el, hl, lgal = samples
    ul_e, ql_e, el_e, hl_e, lgal_e = samples_exact
    var_params, var_params_exact = np.array([]), np.array([])
    for param in ql.T:
        var_params = np.append(var_params, np.var(param))    
    for param in ql_e.T:
        var_params_exact = np.append(var_params_exact, np.var(param))
    ul, ql, el, hl, lgal = samples
    ul_e, ql_e, el_e, hl_e, lgal_e = samples_exact
    _ = plt.plot(var_params, var_params_exact, "." )
    _ = plt.plot([-1,1], [-1,1], "-")
    _ = plt.ylim(0, 2)
    _ = plt.axis([-.001,.007,-.001, .008])
    _ = plt.title(f"Variance parameters of EC in function of HMC")
    _ = plt.xlabel("iteration")
    _ = plt.ylabel("value")
    _ = plt.legend()





