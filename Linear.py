import math
import time
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

class LinearNet(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(LinearNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 256),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, out_feature),
        )

    def forward(self, x):
        return self.layers(x)


def print_time_use(begin_time):
    times = time.time() - begin_time
    time_day = times // 86400
    time_h = (times % 86400) // 3600
    time_m = (times % 3600) // 60
    time_sec = times % 60
    print(f"\rRunning Time: {time_day} Day {time_h} Hours "
          f"{time_m} Min {time_sec:.0f} Sec", end='')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# normal parameters
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 48)')
parser.add_argument('-in', '--input-dim', default=720, type=int,
                    metavar='N', help='input dimensions (default: 720)')
parser.add_argument('-out', '--output-dim', default=10, type=int,
                    metavar='N', help='output dimensions (default: 10)')
parser.add_argument('-work', '--num-workers', default=4, type=int,
                    help='num of dataloader workers')
parser.add_argument('-p', '--pause', default=False, action='store_true',
                    help="Pause in each batch (default: False)")
parser.add_argument('-pt', '--pause-time', default=5, type=int,
                    help="Pause time in each time (default: 5)")
parser.add_argument('-gid', '--gpu-id', default=0, type=int,
                    help="gpu id")
parser.add_argument('-mode', '--mode', default='simulate', type=str,
                    choices=['simulate', 'maximum', 'maximum_single'],
                    help="Warmer mode: simulate, maximum.")
args = parser.parse_args()

print("Initializing...")
model = nn.DataParallel(LinearNet(args.input_dim, args.output_dim)).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 1e-6)

imgs = torch.randn((args.batch_size*2, args.input_dim), dtype=torch.float32)
labels = torch.randint(0, args.output_dim, (args.batch_size*2,))
dataset = torch.utils.data.TensorDataset(imgs, labels)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers)
print("Finish Initialization!")

print("Start warmer at {}".format(time.asctime(time.localtime())))
begin_time = time.time()
while True:
    for img, label in dataloader:
        img = img.cuda()
        label = label.cuda()

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.pause:
            time.sleep(args.pause_time)

        print_time_use(begin_time)
