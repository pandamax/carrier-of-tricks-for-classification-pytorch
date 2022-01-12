import shutil
import sys
sys.path.insert(0,'..')
import torch
from torch.autograd import Variable
from network.mobilenetv2_s import *
import pytorch_to_caffe_git as pytorch_to_caffe
import os
import merge_bn_seq
import merge_bn


if __name__=='__main__':

 
    network = 'mobilenetv2_s'

    net = MobileNetV2_S(num_classes = 2)
    model_pth = '../checkpoint/mobilenetv2_s_RAdam_warmup_cosine_cutmix_labelsmooth_randaug_mixup_2.3M/best_model.pt'

    state_dict = torch.load(model_pth, map_location='cpu')
  
    print(state_dict.keys())
    net.load_state_dict(state_dict)

    net.eval()#make the net as test mode

    # based on zyc's shell
    net = merge_bn._fuse_modules(net)
    # based on tool
    # net = fuse_conv_bn.fuse_bn_recursively(net)
    # net = merge_bn_seq.fuse_bn_recursively(net)

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    #for k,v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    # net.load_state_dict(new_state_dict,False)
    # net = merge_bn_seq.fuse_module(net)
    

    #net.load_state_dict(torch.load(pth_path, map_location='cpu'))
    # net.eval()#make the net as test mode

    input = Variable(torch.ones([1, 3, 32, 32]))
    
    pytorch_to_caffe.trans_net(net, input, network)
    pytorch_to_caffe.save_prototxt('./caffe_model/{}.prototxt'.format(network))
    pytorch_to_caffe.save_caffemodel('./caffe_model/{}.caffemodel'.format(network))
    shutil.copy(model_pth,'./caffe_model/{}'.format(model_pth.split('/')[-1]))
