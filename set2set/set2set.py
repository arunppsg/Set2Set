import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Set2Set(nn.Module):
    """
    The Set2Set model based on `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \mathbf{e}_{i, t} &= f(\mathbf{x}_i, \mathbf{q}_t)

        \\alpha_{i,t} &= \mathrm{softmax}(\mathbf{e}_{i, t})

        \mathbf{r}_t &= \sum_{i=1}^N \\alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Set2Set model is used for learning order invariant representation
    of vectors which can later be used with Seq2Seq models or representing
    vertices/edges of a graph to a feed-forward neural network.

    Args:
        in_channels: The output size of the embedding representing the elements of the set
        processing_steps: The number of steps of computation to perform over the elements of the set 
        num_layers: Number of recurrent layers to use in LSTM

    Inputs:
        x: tensor of shape :math:`(L, N, in_channels)` where L is the number of batches,
            N denotes number of elements in the batch and in_channels is the dimension of
            the element in the batch
    Outputs:
        q: tensor of shape :math:`(L, in_channels)` where L is the number of batches,
            in_channels is the embedding size
    """
    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.out_channels,
                            hidden_size=self.in_channels,
                            num_layers=self.num_layers)

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x):
        """
        Input:
            x: Input of size [batch_size x n x in_channels]
        Output:
            q: Embeding of size [batch_size, out_channels] 
        """
        batch_size = x.size()[0]
        n = x.size()[1]
        hidden = (torch.zeros(self.num_layers, batch_size, self.in_channels),
                  torch.zeros(self.num_layers, batch_size, self.in_channels))
        q_star = torch.zeros(1, batch_size, self.out_channels)

        for i in range(self.processing_steps):
            # q_star: batch_size * out_channels
            q, hidden = self.lstm(q_star, hidden)
            e = torch.einsum("kij,ibj->kib", q, x)
            # e: 1 x batch_size x n
            a = nn.Softmax(dim=2)(e).squeeze(0)
            r = torch.einsum('ij,ijk->ijk', a, x).sum(axis=1)
            # r: 1 x batch_size x n
            q_star = torch.cat([q, r.unsqueeze(0)], dim=-1)
        return q_star.squeeze(0)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
