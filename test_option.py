import argparse

def get_test_args():
    parser = argparse.ArgumentParser()

    # test model.
    parser.add_argument('--model', default='MobileNetV2_S', choices=('ResNet50', 'RegNet', 'EfficientNet','MobileNetV2_S'),
                        help='optimizer to use (ResNet50 | RegNet | EfficientNet)')
    parser.add_argument('--DATASET_DIR', type=str, default= '../DATASETS')

    parser.add_argument('--test_model_path', type=str, default='./checkpoint/xx.pt')
    parser.add_argument('--test_dataset', type=str, default='../test_xxx.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=(32, 32))
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
