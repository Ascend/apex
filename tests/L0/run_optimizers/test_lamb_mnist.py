# Copyright (c) 2021, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function
import argparse
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Optimizer
import apex
from apex import amp

CALCULATE_DEVICE = "npu:0"
SOURCE_DIR = "/home/data/"
EPS = 0.97


def log_lamb_rs(optimizer: Optimizer, event_writer: SummaryWriter, token_count: int):
    """Log a histogram of trust ratio scalars in across layers."""
    results = collections.defaultdict(list)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for i in ('weight_norm', 'adam_norm', 'trust_ratio'):
                if i in state:
                    results[i].append(state[i])

    for k, v in results.items():
        event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, event_writer):
    model.train()
    tqdm_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(tqdm_bar):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.to("cpu").to(torch.float)
        loss = F.nll_loss(output, target)
        loss = loss.to(device).to(torch.float16)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            step = batch_idx * len(data) + (epoch - 1) * len(train_loader.dataset)
            log_lamb_rs(optimizer, event_writer, step)
            event_writer.add_scalar('loss', loss.item(), step)
            tqdm_bar.set_description(
                f'Train epoch {epoch} Loss: {loss.item():.6f}')


def test(args, model, device, test_loader, event_writer: SummaryWriter, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            output = output.to("cpu")
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    event_writer.add_scalar('loss/test_loss', test_loss, epoch - 1)
    event_writer.add_scalar('loss/test_acc', acc, epoch - 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc))
    if acc < EPS:
        raise Exception("Accuracy dose not meet expect!")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam'],
                        help='which optimizer to use')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                        help='weight decay (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--npu', type=int, default=None,
                        help='NPU id to use')
    parser.add_argument('--data', type=str, default=SOURCE_DIR, help='path of dataset')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    global CALCULATE_DEVICE
    if args.npu is not None:
        CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    device = CALCULATE_DEVICE
    print("use ", CALCULATE_DEVICE)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = apex.optimizers.Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999),
                                     adam=(args.optimizer == 'adam'))
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=1024, verbosity=1)
    writer = SummaryWriter()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, writer, epoch)


if __name__ == '__main__':
    main()
