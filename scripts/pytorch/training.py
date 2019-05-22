import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from seisnn.io import get_dir_list
from seisnn.pytorch.dataset import WaveProbDataset
from seisnn.pytorch.model import Nest_Net

pkl_dir = "/mnt/tf_data/dataset/201718select"
pkl_list = get_dir_list(pkl_dir)

split_point = -1000
trainset = WaveProbDataset(pkl_list[:split_point])
trainloader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Nest_Net(in_ch=1, out_ch=1)
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        wave, prob = data
        wave = wave.to(device)
        prob = prob.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(wave)
        loss = criterion(outputs, prob)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

torch.save(model.state_dict(), '/mnt/tf_data/weights/trained_weight.pt')
