'''
test model.

'''
import torch
import os, sys

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH + '/../..')

from network.resnet import *
from network.efficientnet import *
from network.regnet import *
from network.anynet import *
from network.mobilenetv2_s import *
from learning.evaluator import Evaluator
from torchvision import transforms as T
from utils import IsLaneDatasetFromCsvLocation
from test_option import get_test_args


def get_test_model(args, num_classes):
    
    if 'MobileNetV2_S' in args.model:
        model = MobileNetV2_S(num_classes = num_classes)
       
        print("load test model:",args.test_model_path)
        # model = MobileNetV2_S(num_classes = num_classes, checkpoint_dir=args.checkpoint_dir, checkpoint_name=args.checkpoint_name)
        checkpoint = torch.load('{}'.format(args.test_model_path))
        # pdb.set_trace()
        model.load_state_dict(checkpoint)
    else:
        raise NameError('Not Supportes Model')
    
    return model


def make_testdataloader(args):
    
    test_trans = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        ])

    testset = IsLaneDatasetFromCsvLocation(datasets_dir = args.DATASET_DIR, csv_path = args.test_dataset, train=False, transform=test_trans)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return test_loader

def test_split(data_loader, model):
    correct = 0
    total = 0
    pos_total = 0
    neg_total = 0

    pos_correct = 0
    neg_correct = 0
    test_result={}
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            predicted_list = predicted.tolist()
            labels_list = labels.tolist()
            for idx, label in enumerate(labels_list):
                if label == 1:# pos
                    pos_total += 1
                    if predicted_list[idx] == label:
                        pos_correct += 1
                else:
                    neg_total += 1
                    if predicted_list[idx] == label:
                        neg_correct += 1


            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    testdata_num = len(data_loader)
    correct_num = correct              
    test_acc = 100.*correct_num/total
    pos_acc = 100.*pos_correct/pos_total
    neg_acc = 100.*neg_correct/neg_total


    print('----Test Set Results Summary----')
    test_result['total_num'] = testdata_num
    test_result['correct_num'] = correct_num
    test_result['test_acc'] = test_acc
    test_result['pos_acc'] = pos_acc
    test_result['neg_acc'] = neg_acc

    return test_result

def test(data_loader, model):
    correct = 0
    total = 0
    test_result={}
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    testdata_num = len(data_loader)
    correct_num = correct              
    test_acc = 100.*correct_num/total


    print('----Test Set Results Summary----')
    test_result['total_num'] = testdata_num
    test_result['correct_num'] = correct_num
    test_result['test_acc'] = test_acc

    return test_result


def inference_test():
    args = get_test_args()

    model = get_test_model(args, args.num_classes)

    # Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        model = model.cuda()
    test_loader = make_testdataloader(args)

    print('Testing ....')

    # test_result = test(test_loader, model)
    test_result = test_split(test_loader, model)

    print('test_result:',test_result)

if __name__ == '__main__':
    inference_test()
