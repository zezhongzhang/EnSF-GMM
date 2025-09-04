import torch
from torch.distributions.multivariate_normal import MultivariateNormal

"""
Kalman update functions
"""


def KF_update(prior_mean, prior_cov, H, obs_value, obs_cov):
    """
    Pytorch implementation of the analytical Kalman update.

    :param prior_mean: (dim_x)
    :param prior_cov: (dim_x, dim_x)
    :param H: (dim_x, dim_obs) - observation matrix
    :param obs_value: (dim_obs)
    :param obs_cov: (dim_obs, dim_obs)
    :return: posterior mean and covariance matrix
    """

    y_tilde = obs_value - torch.matmul(prior_mean, H)  # (dim_obs)
    S = torch.linalg.multi_dot([H.T, prior_cov, H]) + obs_cov

    # K: (dim_x, dim_obs)
    # K = prior_cov @ H @ np.linalg.inv(S)
    K = torch.linalg.solve(S, prior_cov @ H, left=False)
    # K = torch.linalg.solve(S,  (prior_cov @ H).T).T

    post_mean = prior_mean + torch.matmul(y_tilde, K.T)

    post_cov = prior_cov - torch.linalg.multi_dot([K, H.T, prior_cov])
    return post_mean, post_cov


def EnKF_update(prior_ensemble, obs_fun, obs_value, obs_sigma):
    """
    Pytorch implementation of the EnKF update.

    :param prior_ensemble: (N, dim_x)
    :param obs_fun: (N, dim_x) -> (N, dim_obs)
    :param obs_value: (dim_obs)
    :param obs_sigma: (int) - observation noise standard deviation
    :return: (N, dim_x) - posterior ensemble
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    A = prior_ensemble - torch.mean(prior_ensemble, dim=0)
    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma

    HX = obs_fun(prior_ensemble)
    HA = HX - torch.mean(HX, dim=0)

    P = torch.matmul(HA.T, HA)/(N-1) + torch.eye(dim_obs, device=obs_value.device)*obs_sigma**2

    # P_inv = np.linalg.inv(P)
    # K = (P_inv @ (HA.T @ A))/(N_enkf-1)

    cross_cov = torch.matmul(HA.T, A) / (N-1)
    K = torch.linalg.solve(P, cross_cov)

    ensemble_post = prior_ensemble + torch.matmul(D - HX, K)
    return ensemble_post


def EnKF_update_diag_P(prior_ensemble, obs_fun, obs_value, obs_sigma):
    """
    Pytorch implementation of the EnKF update.

    :param prior_ensemble: (N, dim_x)
    :param obs_fun: (N, dim_x) -> (N, dim_obs)
    :param obs_value: (dim_obs)
    :param obs_sigma: (int) - observation noise standard deviation
    :return: (N, dim_x) - posterior ensemble
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    A = prior_ensemble - torch.mean(prior_ensemble, dim=0)
    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma

    HX = obs_fun(prior_ensemble)
    HA = HX - torch.mean(HX, dim=0)

    P = torch.matmul(HA.T, HA)/(N-1) + torch.eye(dim_obs, device=obs_value.device)*obs_sigma**2


    # make P diagonal
    P_diag = torch.diag(P)

    cross_cov = torch.matmul(HA.T, A) / (N-1)
    # colunm scaling
    K = cross_cov/P_diag[None,:]

    ensemble_post = prior_ensemble + torch.matmul(D - HX, K)
    return ensemble_post


def EnKF_update_direct_sparse_diag(prior_ensemble, obs_dims, obs_value, obs_sigma):
    """
    Pytorch implementation of the EnKF update with sparse direct observations.
        No correlation is used.
    :param prior_ensemble: (N, dim_x)
    :param obs_dims: (dim_obs)
    :param obs_value: (dim_obs)
    :param obs_sigma: (int) - observation noise standard deviation
    :return: (N, dim_x) - posterior ensemble
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma

    HX = prior_ensemble[:, obs_dims]
    prior_var = torch.var(HX, dim=0)

    K = prior_var / (prior_var + obs_sigma**2)

    ensemble_post = prior_ensemble.clone()
    ensemble_post[:, obs_dims] = prior_ensemble[:, obs_dims] + (D - HX)*K
    return ensemble_post


def KF_with_sample(prior_mean, prior_cov, obs_fun, obs_value, obs_sigma, N):
    """
    Pytorch implementation of the KF update with nonlinear observation.

    :param prior_mean: (dim_x)
    :param prior_cov: (dim_x, dim_x)
    :param obs_fun: (N, dim_x) -> (N, dim_obs)
    :param obs_value: (dim_obs)
    :param obs_sigma: (int) - observation noise standard deviation
    :param N: (int) samples for calculation the cross-cov
    :return: posterior mean and covariance matrix
    """
    dim_x = prior_mean.shape[0]
    dim_obs = obs_value.shape[0]

    sample_prior = MultivariateNormal(prior_mean, prior_cov).rsample(torch.Size([N, ]))

    # obs samples
    # hx = obs_fun(sample_prior, H) + np.random.multivariate_normal(np.zeros(dim_obs), obs_cov, size=N)
    hx = obs_fun(sample_prior) # no added noise

    hx_mean = torch.mean(hx, dim=0)
    hx_centered = hx - hx_mean

    # sample cov
    S = torch.matmul(hx_centered.T, hx_centered) / (N-1) + \
        torch.eye(dim_obs, device=obs_value.device)*obs_sigma**2  # (dim_obs, dim_obs)

    # sample cross_cov
    cross_cov = torch.matmul(hx_centered.T, sample_prior - prior_mean) / (N-1)  # (dim_obs, dim_x)

    # kalman gain
    K = torch.linalg.solve(S, cross_cov)  # (dim_obs, dim_x)

    post_mean = prior_mean + torch.matmul(obs_value - hx_mean, K)
    post_cov = prior_cov - torch.matmul(cross_cov.T, K)
    return post_mean, post_cov
