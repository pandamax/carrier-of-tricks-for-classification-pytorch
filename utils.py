import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset  # For custom datasets
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms as T

from network.resnet import *
from network.efficientnet import *
from network.regnet import *
from network.anynet import *
from network.mobilenetv2_s import *

from learning.lr_scheduler import GradualWarmupScheduler
from learning.radam import RAdam
from learning.randaug import RandAugment
import numpy as np
from PIL import Image

class IsLaneDatasetFromCsvLocation(Dataset):
    def __init__(self, datasets_dir, csv_path, train, transform=None):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """
        # Transforms
        # self.to_tensor = transforms.ToTensor()
        # self.net_input_size = (settings.INPUT_W, settings.INPUT_H)
        self.datasets_dir = datasets_dir
        self.train = train
        
        if self.train:
            self.transfroms = transform
        else:
            self.transfroms = transform
        # Read the csv file
        images = []
        labels = []
        instances=[]
        
        with open(csv_path,encoding='gb18030') as f:
            while True:
                data = f.readline()
                data = data.strip()
                if not data:
                    break
                image = data.split('****')[0]
                images.append(image)
                label = int(data.split('****')[1])
                labels.append(label)
                item = self.datasets_dir+image, label
                instances.append(item)
        # print(images)
        # print(labels)
        # pdb.set_trace()
        self.targets=[s[1] for s in instances]
        # First column contains the image paths
        self.image_arr = np.asarray(images)
        # Second column is the labels
        self.label_arr = np.asarray(labels)

        # print(self.image_arr)
        # print(self.label_arr)
        # pdb.set_trace()

        # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(images)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.datasets_dir + self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # check
        # img_as_img.save('test.png')
        # pdb.set_trace()
        # Check if there is an operation
        # some_operation = self.operation_arr[index]
        # If there is an operation
        # if some_operation:
        #     # Do some operation on image
        #     # ...
        #     # ...
        #     pass
        # Transform image to tensor
        # img_as_tensor = self.to_tensor(img_as_img)
        img_as_tensor = self.transfroms(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

def get_model(args, shape, num_classes):
    if 'ResNet' in args.model:
        model = eval(args.model)(
            shape,
            num_classes,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            pretrained=args.pretrained,
            pretrained_path=args.pretrained_path,
            norm=args.norm,
            zero_init_residual=args.zero_gamma
        )#.cuda(args.gpu)
    elif 'RegNet' in args.model:
        model = eval(args.model)(
            shape,
            1000,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name
        )#.cuda(args.gpu)
        pt_ckpt = torch.load('pretrained_weights/RegNetY-1.6GF_dds_8gpu.pyth', map_location="cpu")
        model.load_state_dict(pt_ckpt["model_state"])
        model.head = AnyHead(w_in=model.prev_w, nc=num_classes)#.cuda(args.gpu)
    elif 'EfficientNet' in args.model:
        model = eval(args.model)(
            shape,
            1000,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name
        )#.cuda(args.gpu)
        pt_ckpt = torch.load('pretrained_weights/EN-B2_dds_8gpu.pyth', map_location="cpu")
        model.load_state_dict(pt_ckpt["model_state"])
        model.head = EffHead(w_in=model.prev_w, w_out=model.head_w, nc=num_classes)#.cuda(args.gpu)
    elif 'MobileNetV2_S' in args.model:
        model = MobileNetV2_S(num_classes = num_classes, checkpoint_dir=args.checkpoint_dir, checkpoint_name=args.checkpoint_name)
       
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            print("load pretrained model:",args.pretrained_path)
            # model = MobileNetV2_S(num_classes = num_classes, checkpoint_dir=args.checkpoint_dir, checkpoint_name=args.checkpoint_name)
            checkpoint = torch.load('{}'.format(args.pretrained_path))
            # pdb.set_trace()
            model.load_state_dict(checkpoint)
    else:
        raise NameError('Not Supportes Model')
    
    return model


def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RADAM':
        optimizer_function = RAdam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    else:
        raise NameError('Not Supportes Optimizer')

    kwargs['lr'] = args.learning_rate
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif args.decay_type == 'step_warmup':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler
        )
    elif args.decay_type == 'cosine_warmup':
        cosine_scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=args.epochs//10,
            after_scheduler=cosine_scheduler
        )
    else:
        raise Exception('unknown lr scheduler: {}'.format(args.decay_type))
    
    return scheduler



def make_dataloader(args):
    
    train_trans = T.Compose([
        T.Resize(args.input_size),
        T.RandomHorizontalFlip(0.6),
        T.RandomRotation(10),
        T.RandomVerticalFlip(0.4),
        T.ColorJitter(brightness=1,contrast=1,hue=0.5,saturation=0.5),
        T.ToTensor(),
    ])
    
    if args.randaugment:
        #train_trans.transforms.insert(0, RandAugment(3, 5))
        train_trans.transforms.insert(0, RandAugment(args.rand_n, args.rand_m))

    valid_trans = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        ])
    
    test_trans = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        ])

    #
    # trainset = torchvision.datasets.ImageFolder(root="data/seg_train/seg_train", transform=train_trans)
    # validset = torchvision.datasets.ImageFolder(root="data/seg_train/seg_train", transform=valid_trans)
    # testset = torchvision.datasets.ImageFolder(root="data/seg_test/seg_test", transform=test_trans)
    trainset = IsLaneDatasetFromCsvLocation(datasets_dir = args.DATASET_DIR, csv_path = args.TRAIN_DATASETS, train=True, transform=train_trans)
    validset = IsLaneDatasetFromCsvLocation(datasets_dir = args.DATASET_DIR, csv_path = args.TRAIN_DATASETS, train=False, transform=valid_trans)
    testset = IsLaneDatasetFromCsvLocation(datasets_dir = args.DATASET_DIR, csv_path = args.TEST_DATASETS, train=False, transform=test_trans)

    np.random.seed(args.seed)
    train_idx, valid_idx = train_test_split(np.arange(trainset.data_len), test_size=0.2, shuffle=True, stratify=trainset.targets)
    # targets = trainset.targets
    # train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, valid_loader, test_loader


def plot_learning_curves(metrics, cur_epoch, args):
    x = np.arange(cur_epoch+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4
    plt.legend(lns, ['Train loss', 'Validation loss', 'Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig('{}/{}/learning_curve.png'.format(args.checkpoint_dir, args.checkpoint_name), bbox_inches='tight')
    plt.close('all')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res