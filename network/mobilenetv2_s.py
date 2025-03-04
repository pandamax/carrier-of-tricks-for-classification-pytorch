'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_S(nn.Module):
    

    Width_Multiplier = 2
    cfg = [(1,  4*Width_Multiplier, 1, 1),
            (6,  6*Width_Multiplier, 1, 2),
           (6,  6*Width_Multiplier, 1, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  8*Width_Multiplier, 1, 2),
           (6,  8*Width_Multiplier, 1, 1),
           (6,  24*Width_Multiplier, 1, 2),
           (6,  24*Width_Multiplier, 1, 1),
            (6,  24*Width_Multiplier, 1, 1),
           (6,  80*Width_Multiplier, 1, 1),
    ]

    def __init__(self, num_classes=10, checkpoint_dir='checkpoint', checkpoint_name='MobileNetV2_S'):
        super(MobileNetV2_S, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
   
        self.conv2 = nn.Conv2d(80*2, 80*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(80*8)
        self.linear = nn.Linear(2560, num_classes) # 1280


    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
)
        out = self.layers(out)
      
        out = F.relu(self.bn2(self.conv2(out)))

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out


def test():
    net = MobileNetV2_S(num_classes = 2)
    x = torch.randn(1,3,32,32) #nchw
    y = net(x)
    print('y_size',y.size())
    
if __name__ =='__main__':
    print('======= test ========')
    test()
