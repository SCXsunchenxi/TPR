import torch
from typing import Tuple, Iterable

nn = torch.nn


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def var_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


# taken from https://github.com/ALRhub/rkn_share/ and modified
class SplitDiagGaussianDecoder(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, lod: int, out_dim: int, dec_var_activation: str):
        """ Decoder for low dimensional outputs as described in the paper. This one is "split", i.e., there are
        completely separate networks mapping from latent mean to output mean and from latent cov to output var
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be a vector, images not supported by this decoder)
        :train_conf: configurate dict for training
        """
        self.dec_var_activation = dec_var_activation
        super(SplitDiagGaussianDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim

        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        assert isinstance(self._hidden_layers_mean, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"

        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()
        assert isinstance(self._hidden_layers_var, nn.ModuleList), "_build_hidden_layers_var needs to return a " \
                                                                   "torch.nn.ModuleList or else the hidden weights " \
                                                                   "are not found by the optimizer"

        self._out_layer_mean = nn.Linear(
            in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(
            in_features=num_last_hidden_var, out_features=out_dim)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, latent_mean: torch.Tensor, latent_cov: Iterable[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward pass of decoder
        :param latent_mean:
        :param latent_cov:
        :return: output mean and variance
        """
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)

        if self.dec_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.dec_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.dec_var_activation == 'square':
            var = torch.square(log_var)
        elif self.dec_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.dec_var_activation == 'elup1':
            var = elup1(log_var)
        else: 
            raise Exception('Variance activation function unknown.')
        return mean, var



# taken from https://github.com/ALRhub/rkn_share/ and modified
class BernoulliDecoder(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, lod: int, out_dim: int, args):
        """ Decoder for image output
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be images)
        :param args: parsed arguments
        """
        super(BernoulliDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim

        self._hidden_layers, num_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
            "torch.nn.ModuleList or else the hidden weights " \
            "are not found by the optimizer"
        self._out_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=num_last_hidden, out_channels=1, kernel_size=2, stride=2, padding=5),
                                        nn.Sigmoid())

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, latent_mean: torch.Tensor) \
            -> torch.Tensor:
        """ forward pass of decoder
        :param latent_mean
        :return: output mean
        """
        h_mean = latent_mean
        for layer in self._hidden_layers:
            h_mean = layer(h_mean)
            #print(f'decoder: {h_mean.shape}')
        mean = self._out_layer(h_mean)
        #print(f'decoder mean {mean.shape}')
        return mean
