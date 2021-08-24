# Referencing https://arxiv.org/pdf/1810.00825.pdf
# and the original PyTorch implementation https://github.com/TropComplique/set-transformer/blob/master/blocks.py
import torch
import torch.nn as nn
from models.attention import MultiheadAttention


class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).
        It is invariant to permutations of the
        second dimension of tensor y (`m`).
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SetAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(nn.Module):

    def __init__(self, d, m, h, rff1, rff2):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            rff1, rff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, rff1)
        self.mab2 = MultiheadAttentionBlock(d, h, rff2)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = x.size(0)
        p = self.inducing_points
        p = p.repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))
        self.rff_s = RFF(d)

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # shape [b, k, d]

        # note that in the original paper
        # they return mab(s, rff(z))
        #return self.mab(s, z)
        return self.mab(s, self.rff_s(z))


class SetTransformer(nn.Module):

    def __init__(self, in_dimension, k, out_dimension, only_decoder=False):
        """
        Arguments:
            in_dimension: an integer.
            k: an integer, number of seed vectors.
            out_dimension: an integer.
        """
        super().__init__()
        
        self.only_decoder = only_decoder

        d = 512 # embedding dimension
        #m = 16  # number of inducing points
        h = 4  # number of heads
        
        self.embed = nn.Sequential(
            nn.Linear(in_dimension, d),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            #SetAttentionBlock(d, h, RFF(d)),#InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),#
            SetAttentionBlock(d, h, RFF(d)),#InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),#
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d)),#InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),#
        )
        self.predictor = nn.Sequential(
            nn.Linear(d, out_dimension),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """
        
        x = self.embed(x) # shape [n, d]
        x = x.unsqueeze(0) # shape [b, n, d]
        if not self.only_decoder:
            x = self.encoder(x) # shape [b, n, d]
        x = self.decoder(x) # shape [b, k, d]
        x = x.squeeze(0) # shape [k, d]
        
        return self.predictor(x)


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(inplace=True),
            nn.Linear(d * 4, d), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)