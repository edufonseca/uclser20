from torch import nn
import torch.nn.functional as F
# from IPython import embed

class MLPHead(nn.Module):
    def __init__(self, args, in_channels, mlp_hidden_size):
        super(MLPHead, self).__init__()
        self.args = args

        if self.args["learn"]["head_num"] == 1:
            if self.args["learn"]["head_size"] == 1:
                self.net1 = nn.Linear(in_channels, self.args["learn"]["low_dim"])
            else:
                # usually this one
                self.net1 = nn.Sequential(
                    nn.Linear(in_channels, mlp_hidden_size),
                    nn.BatchNorm1d(mlp_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_hidden_size, self.args["learn"]["low_dim"])
                )

        else:
            if self.args["learn"]["head_size"] == 1:
                self.net1 = nn.Linear(in_channels, self.args["learn"]["low_dim"])
                self.net2 = nn.Linear(self.args["learn"]["low_dim"], self.args["learn"]["low_dim"])
            else:
                self.net1 = nn.Sequential(
                    nn.Linear(in_channels, mlp_hidden_size),
                    nn.BatchNorm1d(mlp_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_hidden_size, self.args["learn"]["low_dim"])
                )
                self.net2 = nn.Sequential(
                    nn.Linear(self.args["learn"]["low_dim"], mlp_hidden_size),
                    nn.BatchNorm1d(mlp_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_hidden_size, self.args["learn"]["low_dim"])
                )

    def forward(self, x):
        if self.args["learn"]["head_num"] == 1:
            x = self.net1(x)
            x = F.normalize(x, p=2, dim=1)
        else:
            x = self.net1(x)
            x = self.net2(x)
            x = F.normalize(x, p=2, dim=1)
        # note the output is L2 normalized
        return x