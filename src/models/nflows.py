import normflows as nf
import torch
from normflows.flows import Flow


class ClampedMLP(nf.nets.MLP):
    def __init__(self, clamp_value=None, **kwargs):
        super().__init__(**kwargs)
        self.clamp_value = clamp_value

    def forward(self, x):
        x = super().forward(x)
        if self.clamp_value is None:
            return x
        x = torch.clamp(x, min=-self.clamp_value, max=self.clamp_value)
        return x


class L2Norm(Flow):
    def __init__(self, eps=1.0e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        norms = torch.norm(z, p=2, dim=1, keepdim=True)
        z_ = z / (norms + self.eps)
        log_det = torch.sum(torch.log(norms + self.eps))
        return z_, log_det


def get_normalizing_flow(network_config, use_gpu):
    normalize_input = network_config.normalize_input
    latent_size = network_config.latent_size
    hidden_size = network_config.hidden_size
    temp = network_config.temp
    if hidden_size is None:
        hidden_size = latent_size * 2
    n_flows = network_config.n_flows
    clamp_value = network_config.clamp_value
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(n_flows):
        s = ClampedMLP(clamp_value=clamp_value,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        t = ClampedMLP(clamp_value=clamp_value,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    if normalize_input:
        flows += [L2Norm()]

    q0 = nf.distributions.DiagGaussian(latent_size)
    # Note that in inverse method which is applied to the features
    # extracted from the backbone, the order of the flows is reversed.
    # ActNorm z-score normalizes (zero mean and unit variance) the input,
    # using two learnable parameters "mean" and "std" which are initialized
    # by the statistics of the first batch.
    nfm = CustomNormalizingFlow(q0=q0, flows=flows, temp=temp)
    return nfm.cuda() if use_gpu else nfm


class CustomNormalizingFlow(nf.NormalizingFlow):
    def __init__(self, q0, flows, p=None, temp=None):
        super().__init__(q0=q0, flows=flows, p=p)
        self.temp = temp

    def get_sample(self, num_samples):
        self.q0.temperature = self.temp
        # print(self.q0.temperature)
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z  # , log_q
