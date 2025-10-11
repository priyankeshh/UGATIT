import argparse
import torch
import numpy as np
import os
import random
from train import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='UGATIT',
                        help="name your experiment")
    parser.add_argument(
        '--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--gpus', type=int, default=1,
                        help='how many GPUs in one node')
    parser.add_argument('--model_type', type=str, default='normal')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument(
        '--datasetATrain', help="Path to domain A train dataset")
    parser.add_argument(
        '--datasetBTrain', help="Path to domain B train dataset")
    parser.add_argument(
        '--datasetATest', help="Path to domain A Test dataset")
    parser.add_argument(
        '--datasetBTest', help="Path to domain B test dataset")

    args = parser.parse_args()

    if args.model_type == 'normal':
        from configs.train import get_cfg
    elif args.model_type == 'light':
        from configs.light import get_cfg

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    cfg = get_cfg()
    cfg.update(vars(args))
    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    model = Model(cfg)
    print(
        f"------------------ Starting Training for {args.name} ------------------")
    model.train()
