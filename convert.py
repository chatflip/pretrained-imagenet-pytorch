import argparse

import torch
import torch.nn as nn
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-path', default='~/Downloads/ilsvrc2012', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if 'mobilenetv2' in args.model:
        root = 'mobilenetv2'
    elif 'inceptionv3' in args.model:
        root = 'inceptionv3'
    src_name = '{}/model_{}.pth'.format(root, args.epochs)

    param = torch.load(src_name)

    if args.model == 'mobilenetv2_width_multi13':
        model = torchvision.models.mobilenet_v2(width_mult=1.3)
        dst_name = '{0}/mobilenetv2_13_imagenet_res224.pth'.format(root)
    elif args.model == 'inceptionv3_res224':
        model = torchvision.models.inception_v3(
            pretrained=True, aux_logits=False, transform_input=False)
        dst_name = '{0}/{0}_imagenet_res224.pth'.format(root)
    else:
        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    model.load_state_dict(param['model'])
    torch.save(model.state_dict(), dst_name)