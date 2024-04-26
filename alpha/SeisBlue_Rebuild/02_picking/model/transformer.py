# -*- coding: utf-8 -*-
from torch import nn, cat, optim
import torch
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class TransformerModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dim_model=512, num_heads=8,
                 num_encoder_layers=8, dim_feedforward=2048,
                 max_seq_length=5000, dropout=0.1):
        super(TransformerModel, self).__init__()
        assert dim_model % 2 == 0, "dim_model must be divisible by 2"

        self.conv1 = nn.Conv1d(in_channels, dim_model // 2, kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(dim_model // 2, dim_model, kernel_size=3,
                               padding=1)
        self.conv_norm = nn.BatchNorm1d(dim_model)
        self.conv_activation = nn.ReLU()

        # Embedding layer for input
        self.input_embedding = nn.Linear(dim_model, dim_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(dim_model, dropout,
                                                      max_seq_length)

        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                        nhead=num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_encoder_layers)

        # Final linear layer
        self.output_linear = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        # CNN layers
        x = self.conv_activation(self.conv_norm(self.conv1(x)))
        x = self.conv_activation(self.conv_norm(self.conv2(x)))

        # Convert input to (seq_len, batch, features)
        x = x.permute(2, 0, 1)

        # Input embedding and positional encoding
        x = self.input_embedding(x) * math.sqrt(self.dim_model)
        x = self.positional_encoding(x)

        # Transformer with checkpointing
        for layer in self.transformer_encoder.layers:
            x = checkpoint(layer, x)

        # Convert back to (batch, seq_len, features)
        x = x.permute(1, 0, 2)

        # Final linear layer
        x = self.output_linear(x)

        return F.log_softmax(x, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
            self, in_channels=3, classes=3, sampling_rate=100, **kwargs
    ):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.classes = classes
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=11, padding='same')
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding='same')
        self.conv3 = nn.Conv1d(16, 16, kernel_size=7, padding='same')
        self.conv4 = nn.Conv1d(16, 32, kernel_size=5, padding='same')
        self.conv5 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.resnet = self.build_resnet(64, 3)


        self.bilstm1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True,
                               bidirectional=True)
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=1, padding='same')
        self.bilstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True,
                               bidirectional=True)
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=64,
                               kernel_size=1, padding='same')
        self.dropout = nn.Dropout(0.1)

        self.layernorm = nn.LayerNorm(normalized_shape=[64, 64],
                                      elementwise_affine=True)

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True,
                            bidirectional=False)

        self.transformer_block = TransformerEncoder(embed_dim=64,
                                                    num_heads=2,
                                                    ff_dim=1024)
        self.transformer_blockE = TransformerEncoder(embed_dim=64,
                                                     num_heads=2,
                                                     ff_dim=1024)

        self.up1N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9N = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.up2N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10N = nn.Conv1d(96, 96, kernel_size=5, padding=2)
        self.up3N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11N = nn.Conv1d(96, 32, kernel_size=5, padding=2)
        self.up4N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv12N = nn.Conv1d(32, 16, kernel_size=7, padding=3)
        self.up5N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13N = nn.Conv1d(16, 16, kernel_size=9, padding=4)
        self.up6N = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv14N = nn.Conv1d(16, 1, kernel_size=11, padding=5)

        self.lstmP = nn.LSTM(input_size=64, hidden_size=64, batch_first=True,
                             bidirectional=False)
        self.transformer_blockP = TransformerEncoder(embed_dim=64,
                                                     num_heads=2,
                                                     ff_dim=1024)
        self.up1P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9P = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.up2P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10P = nn.Conv1d(96, 96, kernel_size=5, padding=2)
        self.up3P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11P = nn.Conv1d(96, 32, kernel_size=5, padding=2)
        self.up4P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv12P = nn.Conv1d(32, 16, kernel_size=7, padding=3)
        self.up5P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13P = nn.Conv1d(16, 16, kernel_size=9, padding=4)
        self.up6P = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv14P = nn.Conv1d(16, 1, kernel_size=11, padding=5)

        self.lstmS = nn.LSTM(input_size=64, hidden_size=64, batch_first=True,
                             bidirectional=False)
        self.transformer_blockS = TransformerEncoder(embed_dim=64,
                                                     num_heads=2,
                                                     ff_dim=1024)
        self.up1S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9S = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.up2S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10S = nn.Conv1d(96, 96, kernel_size=5, padding=2)
        self.up3S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11S = nn.Conv1d(96, 32, kernel_size=5, padding=2)
        self.up4S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv12S = nn.Conv1d(32, 16, kernel_size=7, padding=3)
        self.up5S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13S = nn.Conv1d(16, 16, kernel_size=9, padding=4)
        self.up6S = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv14S = nn.Conv1d(16, 1, kernel_size=11, padding=5)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = self.activation(self.conv4(x))
        x = self.pool(x)
        x = self.activation(self.conv5(x))
        x = self.pool(x)
        x = self.activation(self.conv6(x))
        x = self.pool(x)
        x = self.resnet(x)

        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.activation(self.conv7(x))
        x = x.transpose(1, 2)

        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.activation(self.conv8(x))
        x = x.transpose(1, 2)
        x = self.layernorm(x)

        x, _ = self.lstm(x)
        x = self.transformer_block(x)
        transE = self.transformer_blockE(x)

        xN = self.up1N(transE)
        xN = self.activation(self.conv9N(xN))
        xN = self.up2N(xN)
        xN = self.activation(self.conv10N(xN))
        xN = self.up3N(xN)
        xN = self.activation(self.conv11N(xN))
        xN = self.up4N(xN)
        xN = self.activation(self.conv12N(xN))
        xN = self.up5N(xN)
        xN = self.activation(self.conv13N(xN))
        xN = torch.sigmoid(self.conv14N(xN))

        xP, _ = self.lstmP(transE)
        xP = self.transformer_blockP(xP)
        xP = self.up1P(xP)
        xP = self.activation(self.conv9P(xP))
        xP = self.up2P(xP)
        xP = self.activation(self.conv10P(xP))
        xP = self.up3P(xP)
        xP = self.activation(self.conv11P(xP))
        xP = self.up4P(xP)
        xP = self.activation(self.conv12P(xP))
        xP = self.up5P(xP)
        xP = self.activation(self.conv13P(xP))
        xP = torch.sigmoid(self.conv14P(xP))

        xS, _ = self.lstmS(transE)
        xS = self.transformer_blockS(xS)
        xS = self.up1S(xS)
        xS = self.activation(self.conv9S(xS))
        xS = self.up2S(xS)
        xS = self.activation(self.conv10S(xS))
        xS = self.up3S(xS)
        xS = self.activation(self.conv11S(xS))
        xS = self.up4S(xS)
        xS = self.activation(self.conv12S(xS))
        xS = self.up5S(xS)
        xS = self.activation(self.conv13S(xS))
        xS = torch.sigmoid(self.conv14S(xS))

        output = torch.concatenate([xP, xS, xN], axis=1)

        return output

    def build_resnet(self, filter_nums, block_nums, strides=1):
        layers = []
        layers.append(ResBlock(filter_nums, strides))
        for _ in range(1, block_nums):
            layers.append(ResBlock(filter_nums, 1))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, filter_nums, strides):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filter_nums, filter_nums, kernel_size=3,
                               stride=strides, padding=1)
        self.conv2 = nn.Conv1d(filter_nums, filter_nums, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(filter_nums)
        self.bn2 = nn.BatchNorm1d(filter_nums)
        self.drop = nn.Dropout(0.1)
        if strides != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(filter_nums, filter_nums, 1, stride=strides)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.drop(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.conv2(out)

        out += identity
        out = F.relu(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=rate,
            activation='relu'
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        transformer_output = self.transformer_encoder_layer(inputs)
        output = self.layernorm(inputs + transformer_output)

        return output


if __name__ == "__main__":
    pass
