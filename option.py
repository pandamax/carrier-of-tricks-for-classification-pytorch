import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # model architecture & checkpoint
    parser.add_argument('--model', default='MobileNetV2_S', choices=('ResNet50', 'RegNet', 'EfficientNet','MobileNetV2_S'),
                        help='optimizer to use (ResNet50 | RegNet | EfficientNet)')
    parser.add_argument('--norm', default='batchnorm', choices=('batchnorm', 'evonorm'),
                        help='normalization to use (batchnorm | evonorm)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default='./checkpoint/mobilenetv2_s_RAdam_warmup_cosine_cutmix_labelsmooth_randaug_mixup_2.3M/best_model.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='')
    parser.add_argument('--zero_gamma', action='store_true', default=False)

    # data loading
    parser.add_argument('--DATASET_DIR', type=str, default= '../DATASETS')
    parser.add_argument('--TRAIN_DATASETS', type=str, default='../train_xxx.txt')
    parser.add_argument('--TEST_DATASETS', type=str, default='../test_xxx.txt')
    parser.add_argument('--input_size', type=int, default=(32, 32))

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # training hyper parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha')
    parser.add_argument('--label_smooth', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--cutmix_alpha', type=float, default=0.0, help='cutmix alpha')
    parser.add_argument('--cutmix_prob', type=float, default=0.0, help='cutmix probability')
    parser.add_argument('--randaugment', action='store_true', default=False)
    parser.add_argument('--rand_n', type=int, default=3)
    parser.add_argument('--rand_m', type=int, default=15)

    # optimzier & learning rate scheduler
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'ADAM', 'RADAM'),
                        help='optimizer to use (SGD | ADAM | RADAM)')
    parser.add_argument('--decay_type', default='step', choices=('step', 'step_warmup', 'cosine_warmup'),
                        help='optimizer to use (step | step_warmup | cosine_warmup)')

    args = parser.parse_args()
    return args
