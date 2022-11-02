import torch
from lib.CRUCell import RKNCell, CRUCell
nn = torch.nn


# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRULayer(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, latent_obs_dim, args, dtype=torch.float64):
        super().__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._cell = RKNCell(latent_obs_dim, args, dtype) if args.rkn else CRUCell(latent_obs_dim, args, dtype)

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, latent_obs, obs_vars, initial_mean, initial_cov, obs_valid=None, time_points=None):
        """Passes the entire observation sequence sequentially through the Kalman component

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating if observation is valid 
        :param time_points: timestamp of the observation
        """

        # prepare list for return
        prior_mean_list = []
        prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]
        kalman_gain_list = [[], []]

        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov
        T = latent_obs.shape[1]

        # iterate over sequence length
        for i in range(T):
            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
            delta_t = time_points[:, i+1] - time_points[:,
                                                        i] if time_points is not None and i < T-1 else torch.ones_like(latent_obs)[:, 0, 0]
            post_mean, post_cov, next_prior_mean, next_prior_cov, kalman_gain = \
                self._cell(prior_mean, prior_cov,
                           latent_obs[:, i], obs_vars[:, i], cur_obs_valid, delta_t=delta_t)
            #print(f'post_mean {post_mean.shape}, next_prior_mean {next_prior_mean.shape}')

            post_mean_list.append(post_mean)
            [post_cov_list[i].append(post_cov[i]) for i in range(3)]
            prior_mean_list.append(next_prior_mean)
            [prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]
            [kalman_gain_list[i].append(kalman_gain[i]) for i in range(2)]

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = [torch.stack(x, 1) for x in prior_cov_list]
        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]
        kalman_gains = [torch.stack(x, 1) for x in kalman_gain_list]

        return post_means, post_covs, prior_means, prior_covs, kalman_gains
