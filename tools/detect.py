# import numpy as np
# import torch
# import cv2
# import os
# import os.path as osp
# import glob
# import argparse
# from lanedet.datasets.process import Process
# from lanedet.models.registry import build_net
# from lanedet.utils.config import Config
# from lanedet.utils.visualization import imshow_lanes
# from lanedet.utils.net_utils import load_network
# from pathlib import Path
# from tqdm import tqdm

from lanedet.detect import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
