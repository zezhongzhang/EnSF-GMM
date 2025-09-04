import torch
import numpy as np

# the lorenz drift
def lorenz96_drift(x, t):
    return (torch.roll(x, -1) - torch.roll(x, 2))*torch.roll(x, 1) - x + 8

def lorenz96_drift_np(x, t):
    return (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - x + 8

def PF_resample(prior_sample_pf, log_likelihood_fn, boot_size):
    N_pf = prior_sample_pf.shape[0]
    log_l = log_likelihood_fn(prior_sample_pf)
    log_l -= torch.max(log_l)
    l_pf = torch.exp(log_l)
    p = l_pf / torch.sum(l_pf)
    idx = np.random.choice(N_pf, boot_size, p=p.cpu().numpy(), replace=True)
    post_sample_pf = prior_sample_pf[idx, :]
    return post_sample_pf

def state_clip(x_state, tol):
    in_range = x_state.abs().max(dim=1)[0] <= tol
    num_in_range = torch.sum(in_range).item()

    if num_in_range == 0:
        return None
    else:
        num_out_range = x_state.shape[0] - num_in_range
        # pick = torch.randperm(num_in_range)[:num_out_range]
        pick = torch.randint(low=0,high=num_in_range, size=(num_out_range,))

        x_state_in = x_state[in_range, :]
        # replace
        x_state[~in_range,:] = x_state_in[pick,:]
        return num_out_range


def state_clip_np(x_state, tol):
    in_range = x_state.abs().max(dim=1)[0] <= tol
    num_in_range = torch.sum(in_range).item()

    if num_in_range == 0:
        return None
    else:
        num_out_range = x_state.shape[0] - num_in_range
        # pick = torch.randperm(num_in_range)[:num_out_range]
        pick = torch.randint(low=0,high=num_in_range, size=(num_out_range,))

        x_state_in = x_state[in_range, :]
        # replace
        x_state[~in_range,:] = x_state_in[pick,:]
        return num_out_range


class IntegratorODE:
    def __init__(self, drift_fun=None):
        self.drift_fun = drift_fun

    def integrate(self, xt, t_start, t_end, num_steps, method='fe', save_path=False):
        """
        Solving a ODE from t_start to t_end with a batch of inputs
        """
        if self.drift_fun is None:
            raise ValueError('Drift function not specified!')

        dt = (t_end - t_start) / num_steps
        t_current = t_start

        path_all = []
        t_vec = []

        # saving the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t_current)

        for i in range(num_steps):
            xt = self.integrate_one_step(xt, t_current, dt, method)
            t_current = t_current + dt
            # saving the path
            if save_path:
                path_all.append(xt)
                t_vec.append(t_current)

        if save_path:
            return xt, [path_all, t_vec]
        else:
            return xt

    def integrate_one_step(self, xt, t_current, dt, method):
        """
        perform one-step update with specified method
        """
        if method == 'rk4':
            return self.rk4(xt=xt, f=self.drift_fun, t=t_current, dt=dt)
        elif method == 'fe':
            return self.forward_euler(xt=xt, f=self.drift_fun, t=t_current, dt=dt)
        else:
            raise NotImplementedError('Not implemented!')

    @staticmethod
    def forward_euler(xt, f, t, dt):
        xt = xt + f(xt, t) * dt
        return xt

    @staticmethod
    def rk4(xt, f, t, dt):
        k1 = f(xt, t)
        k2 = f(xt + dt / 2 * k1, t + dt / 2)
        k3 = f(xt + dt / 2 * k2, t + dt / 2)
        k4 = f(xt + dt * k3, t + dt)
        return xt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def set_drift_fun(self, drift_fun):
        self.drift_fun = drift_fun


def rmse_torch(x, y):
    return torch.sqrt(torch.mean( (x-y)**2)).item()

def rmse_np(x, y):
    return np.sqrt(np.mean( (x-y)**2))

def rk4(xt, fn, t, dt):
    k1 = fn(xt, t)
    k2 = fn(xt + dt / 2 * k1, t + dt / 2)
    k3 = fn(xt + dt / 2 * k2, t + dt / 2)
    k4 = fn(xt + dt * k3, t + dt)
    return xt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
