# from models.mobilenetv2_b import MobileNetV2_B
# from models.mobilenetv2_s import MobileNetV2
# from models.mobilenetv2 import MobileNetV2
from network.mobilenetv2_s import MobileNetV2_S
import sys
import os
import torch
sys.path.insert(0,'.')
import pdb
# from models import *

class_num = 2

B = 1
C = 3
H = 32
W = 32

print('==== flops ====')
dummy_input = torch.randn(B, C, H, W, device='cpu') #(B,C,H,W)-(10,3,224,224)
# model =  MobileNetV2(num_classes = class_num)
model =  MobileNetV2_S(num_classes = class_num)


from torchsummaryX import summary
summary(model, torch.zeros((B, C, H, W)))

from thop import clever_format
import pdb
from thop import profile
input = torch.randn(B, C, H, W)
macs, params = profile(model, inputs=(input, ))
print(macs, params)
macs, params = clever_format([macs, params], "%.3f")
print('macs:{}\nparams:{}'.format(macs, params))
print("FINISHED!")

print("finish")

#####################
# macs:2.558G
# params:2.479M
