# -*- coding: utf-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
import seisblue.model


def plot_tensor_board(model):
    batch_size = 10
    in_channels = 1
    in_samples = 256
    input_data = torch.randn(batch_size, in_channels, in_samples)

    writer = SummaryWriter('./log')
    writer.add_graph(model, input_to_model=input_data)
    writer.close()


if __name__ == '__main__':
    model = seisblue.model.FMNet()
    plot_tensor_board(model)
