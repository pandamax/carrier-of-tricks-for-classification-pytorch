'''
用于对比原pytorch模型与fuse_bn后且转为caffemodel的前向主观测试
'''
import caffe
import numpy as np  
import os
import sys
sys.path.insert(0,'..')
import cv2
from copy import deepcopy
import pdb
from network.mobilenetv2_s import *
import torch

#set parameters.
class param:
    def __init__(self):
        self.color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
                    (100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]

        self.num_classes = 2
        self.input_size =(32, 32)
        self.pytorch_model_path = './caffe_model/best_model.pt'
        self.caffe_model_prototxt_path = './caffe_model/mobilenetv2_s.prototxt'
        self.caffe_model_caffemodel_path = './caffe_model/mobilenetv2_s.caffemodel'
        self.test_images = './imgs'

p = param() 

def to_np(test_image):
    test_image = np.rollaxis(test_image, axis=2, start=0)
    inputs = test_image.astype(np.float32)
    inputs = inputs[np.newaxis,:,:,:] 
    return inputs

def load_caffemodel():
    caffe.set_mode_cpu()
    protofile =  p.caffe_model_prototxt_path
    weightfile = p.caffe_model_caffemodel_path 
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    return net

# Reference from:
def caffemodel_out(net, image):
    net.blobs['data'].data[...] = image
    output = net.forward()
    # return [output['conv_blob113'],output['conv_blob116'],output['conv_blob119']]
    return output


def load_pytorchmodel():
    net =  MobileNetV2_S(p.num_classes)
    state_dict = torch.load(p.pytorch_model_path, map_location='cpu')
    net.load_state_dict(state_dict)
    return net

def pytorchmodel_out(net, image):
    image = torch.from_numpy(image).float() 
    with torch.no_grad():
        result= net(image)
        # pdb.set_trace()

    return result
    
def inference(test_images):
    
    caffe_net = load_caffemodel()

    pytorch_net = load_pytorchmodel()


    img_list = os.listdir(test_images)
    # img_list = [img for img in img_list if '.jpg' in img]
    img_list = [img for img in img_list if 'png' in img or '.jpg' in img]

    for img in img_list:
        print("Now Dealing With:",img)
        ori_image = cv2.imread(test_images + '/' + img) #hw, cv2.IMREAD_UNCHANGED

        # caffe_img = ori_image.copy()
        # pytorch_img = ori_image.copy()
        # print("ori_image shape:",ori_image.shape)
        test_image = cv2.resize(ori_image, p.input_size)/255
        
        test_image = to_np(test_image)

        pred_pytorch = pytorchmodel_out(pytorch_net.eval(), test_image)

        pred_caffe = caffemodel_out(caffe_net, test_image)

        print('pred_pytorch:{}\npred_caffe:{}'.format(pred_pytorch, pred_caffe))
        

        
       
if __name__ == '__main__':

    inference(p.test_images)
    print("finished")


