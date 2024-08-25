import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
from lanedet.models.nets import Detector
import onnxruntime as ort
from torch import nn
from lanedet.core.lane import Lane
#configs/laneatt/mobilenetv2_tusimple_200epochs.py --savedir vis/bla --load_from models/mobilenet_tusimple_200epochs.pth --img data/Labor/racecar_image_1.jpg
class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            print(data)
            data = self.net(data)
            print(data)
            data = self.net.module.get_lanes(data)
            print(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def proposals_to_pred(proposals):
    IMG_W = 640
    IMG_H = 360
    S = 72
    n_offsets = S
    n_strips = S - 1
    anchor_ys = torch.linspace(1, 0, steps=n_offsets, dtype=torch.float32).cuda()
    anchor_ys = anchor_ys.to(proposals.device)
    anchor_ys = anchor_ys.double()
    lanes = []
    for lane in proposals:
        lane_xs = lane[5:] / IMG_W
        start = int(round(lane[2].item() * n_strips))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        # end = label_end
        # if the proposal does not start at the bottom of the image,
        # extend its proposal until the x is outside the image
        mask = ~((((lane_xs[:start] >= 0.) &
                    (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = anchor_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = lane_xs.flip(0).double()
        lane_ys = lane_ys.flip(0)
        if len(lane_xs) <= 1:
            continue
        points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
        lane = Lane(points=points.cpu().numpy(),
                    metadata={
                        'start_x': lane[3],
                        'start_y': lane[2],
                        'conf': lane[1]
                    })
        lanes.append(lane)
    return lanes

def get_lanes(output, as_lanes=True):
    proposals_list = output['proposals_list']
    softmax = nn.Softmax(dim=1)
    decoded = []
    for proposals, _, _, _ in proposals_list:
        proposals[:, :2] = softmax(proposals[:, :2])
        proposals[:, 4] = torch.round(proposals[:, 4])
        if proposals.shape[0] == 0:
            decoded.append([])
            continue
        if as_lanes:
            pred = proposals_to_pred(proposals)
        else:
            pred = proposals
        decoded.append(pred)
    return decoded

def process(args):
    cfg = Config.fromfile(args.config)
    processes = Process(cfg.val_process, cfg)
    cvimg = cv2.imread("zeros.jpg").astype(np.float32)
    down_points = (640, 360)

    # cvimg = cv2.resize(cvimg, down_points, interpolation= cv2.INTER_LINEAR)
    # input_img = np.asanyarray(cvimg, dtype=np.float32).reshape(1, 3, 360, 640)
    data = {'img': cvimg, 'lanes': []}
    data = processes(data)
    data['img'] = np.asanyarray(data['img'].unsqueeze(0), dtype=np.float32)
    ort_session = ort.InferenceSession('models/mobilenetv2_model_200epochs.onnx',providers=["CUDAExecutionProvider"])
    # compute ONNX Runtime output prediction
    detector = Detector(cfg, backbone=ort_session)
    output = detector(data)
    lanes = get_lanes(output)

    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

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


