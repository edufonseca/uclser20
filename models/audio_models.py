import torch
import torch.nn as nn


class Conv_bn_relu_mp_2d(nn.Module):
    def __init__(self, in_channels, out_channels, shape=(3, 3), stride=1, pooling=(2, 2)):
        super(Conv_bn_relu_mp_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, shape, stride=stride, padding=shape[0] // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mpool(self.relu(self.bn(self.conv(x))))
        return out


class Conv_bn_relu_2d(nn.Module):
    def __init__(self, in_channels, out_channels, shape=(3, 3), stride=1):
        super(Conv_bn_relu_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, shape, stride=stride, padding=shape[0] // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class CRNN_tagger(nn.Module):
    '''
    Inspired by Cakir et al.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7933050
    extract local features with conv stack & GRU to model temporal structure
    However, we do tagging (not SED) hence, we want GRU to return one single vector summarizing time (not a sequence)

    tensor format: batch, channels, time, freq
    '''

    def __init__(self, args, num_classes=20):
        super(CRNN_tagger, self).__init__()

        self.args = args

        self.conv1 = Conv_bn_relu_mp_2d(1, 128, shape=(5, 5), pooling=(2, 5))
        self.conv2 = Conv_bn_relu_mp_2d(128, 128, shape=(5, 5), pooling=(2, 4))
        self.conv3 = Conv_bn_relu_mp_2d(128, 128, shape=(5, 5), pooling=(2, 2))

        self.rnn1 = nn.GRU(input_size=256, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        # output dense
        self.dense = nn.Linear(64, num_classes)

    def forward(self, x):
        # input is time-freq patches of 101x96

        # Conv stack to extract local features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 1) Reorder (batch, time, freq, channel)
        x = x.permute(0, 2, 3, 1)
        # 2) Reshape to (the original time dimension, x.size(1), whatever is needed to match dimensions)
        x = x.reshape(x.size(0), x.size(1), -1)
        # batch, seq, feature

        x, _ = self.rnn1(x)
        # x is output of shape (batch, seq_len num_directions * hidden_size)

        # the directions can be separated using output.view(),
        # with forward and backward being direction 0 and 1 respectively.
        # x.view(batch, seq_len, num_directions, hidden_size)
        x = x.view(x.size(0), x.size(1), 2, 64)
        # take the output features for the last timestep of the sequence, for forward direction
        # -1 for the last step, 0/1 id the direction
        forw = x[:, -1, 0, :]
        backw = x[:, -1, 1, :]

        # emulate the merge_mode = 'mul', where outputs of the forward and backward RNNs will be combined by element-wise multiply
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
        x = torch.mul(forw, backw)

        # output dense
        if self.args["learn"]["method"] == "Contrastive":
            x1 = self.dense(x)
            return x1, x
        else:
            x = self.dense(x)
            # do not include activation
            return x


class VGGlike_small_emb_relu(nn.Module):
    '''
    tensor format: batch, channels, time, freq

    VGG321: conv32 x3, MP, conv64x2, MP, conv128x1, MP. Then global pooling and relu(dense) + dense

    the embeddings to input the projection head are the output of RELU after dense1
    '''

    # tensor format: batch, channels, time, freq
    def __init__(self, args, num_classes=20):
        super(VGGlike_small_emb_relu, self).__init__()
        self.args = args
        self.conv0 = Conv_bn_relu_2d(1, 32, shape=(3, 3))
        self.conv1 = Conv_bn_relu_2d(32, 32, shape=(3, 3))
        self.conv2 = Conv_bn_relu_mp_2d(32, 32, shape=(3, 3), pooling=(2, 2))
        self.conv3 = Conv_bn_relu_2d(32, 64, shape=(3, 3))
        self.conv4 = Conv_bn_relu_mp_2d(64, 64, shape=(3, 3), pooling=(2, 2))
        self.conv5 = Conv_bn_relu_mp_2d(64, 128, shape=(3, 3), pooling=(2, 2))

        # output dense
        embed_size = args["learn"]["embed_size"]
        if self.args["learn"]["global_pooling"] == "gapgmp":
            in_channels_dense1 = 256
        else:
            in_channels_dense1 = 128
        self.dense1 = nn.Linear(in_channels_dense1, embed_size)
        self.relu = nn.ReLU()
        self.dense_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # input is time-freq patches of 101x96
        # tensor format: batch, channels, time, freq

        # Conv stack
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.args["learn"]["global_pooling"] == "gap":
            # tensor format: batch, channels, time, freq
            gap = nn.AdaptiveAvgPool2d((1, 1))(x)
            pool_summary = gap.view(-1, 128)

        elif self.args["learn"]["global_pooling"] == "gapgmp":
            # tensor format: batch, channels, time, freq
            gmp = nn.AdaptiveMaxPool2d((1, 1))(x)
            gap = nn.AdaptiveAvgPool2d((1, 1))(x)
            pool_gmpgap = torch.cat([gmp, gap], dim=1)
            pool_summary = pool_gmpgap.view(-1, 256)

        # dense and output dense
        if self.args["learn"]["method"] == "Contrastive":
            out1 = self.dense1(pool_summary)
            out_emb = self.relu(out1)
            out1 = self.dense_out(out_emb)

            # the embeddings to feed the projection head are the output of ReLU after dense1
            return out1, out_emb

        else:
            out = self.dense1(pool_summary)
            out = self.relu(out)
            out = self.dense_out(out)
            # do not include activation
            return out
