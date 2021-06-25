import math
import time
import torch
import argparse
import torchvision
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms


class ResNet2(nn.Module):

    def __init__(self, block, layers, fcExpansion, num_classes=1000):
        self.inplanes = 64
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * fcExpansion, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

def print_time_use(begin_time):
    times = time.time() - begin_time
    time_day = times // 86400
    time_h = (times % 86400) // 3600
    time_m = (times % 3600) // 60
    time_sec = times % 60
    print(f"\rRunning Time: {time_day} Day {time_h} Hours {time_m} Min {time_sec:.0f} Sec", end='')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# normal parameters
parser.add_argument('-s', '--single-gpu', default=False, action='store_true',
                    help='warmer with single gpu')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 24/10G)')
parser.add_argument('-img', '--image-size', default=720, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-num', '--sample-number', default=2560, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-cls', '--num-class', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-work', '--num-workers', default=4, type=int,
                    help='num of dataloader workers')
parser.add_argument('-p', '--pause', default=False, action='store_true',
                    help="Pause in each batch (default: False)")
parser.add_argument('-pt', '--pause-time', default=5, type=int,
                    help="Pause time in each time (default: 5)")
parser.add_argument('-mode', '--mode', default='simulate', type=str, choices=['simulate', 'maximum'],
                    help="Warmer mode: simulate, maximum.")
args = parser.parse_args()

print("Initializing...")
model = nn.DataParallel(ResNet2(BasicBlock, [3, 4, 6, 3], fcExpansion=289, num_classes=args.num_class)).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 1e-6, momentum=1e-4, weight_decay=1e-4)

if args.mode == 'simulate':
    imgs = torch.randn((args.sample_number, 3, args.image_size, args.image_size), dtype=torch.float32)
    labels = torch.randint(0, args.num_class, (args.sample_number,))
    dataset = torch.utils.data.TensorDataset(imgs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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

elif args.mode == 'maximum':
    imgs = torch.randn((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32).cuda()
    labels = torch.randint(0, args.num_class, (args.sample_number,)).cuda()
    print("Finish Initialization!")

    print("Start warmer at {}".format(time.asctime(time.localtime())))
    begin_time = time.time()
    i = 0
    while True:
        output = model(imgs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
        if i == 100:
            i = 0
            print_time_use(begin_time)
else:
    AttributeError("No such mode: {}".format(args.mode))
