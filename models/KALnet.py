import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


class KAL_Net(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, polynomial_order=3):
        super(KAL_Net, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = polynomial_order

        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for Legendre expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Initialize network parameters
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):  # 进行迭代表达式
            # Base weight for linear transformation in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Polynomial weight for handling Legendre polynomial expansions
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            # Layer normalization to stabilize learning and outputs
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]
        # legendre_polys =[p0明文,p1密文,p2明文x密文的生成式（矩阵乘法）]
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        # Ensure x is on the right device from the start, matching the model parameters
        x = x.to(self.base_weights[0].device)

        for i, (base_weight, poly_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.poly_weights, self.layer_norms)):
            # Apply base activation to input and then linear transform with base weights
            base_output = F.linear(x * x, base_weight)  # 使用 x^2 作为激活函数  2

            # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1 # 可以先用明文进行归一化
            # Compute Legendre polynomials for the normalized x
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order) # 4
            # Reshape legendre_basis to match the expected input dimensions for linear transformation
            legendre_basis = legendre_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(legendre_basis, poly_weight) #密文 x 明文 # 8
            # Combine base and polynomial outputs, normalize, and activate
            x = base_output + poly_output
            x = layer_norm(x)
            x = x * x  # 使用 x^2 作为激活函数 # 密文进行了一次 # 16

        return x