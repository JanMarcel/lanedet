import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from lanedet.utils.linklabel_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


@DATASETS.register_module
class LinkLabel(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = [cfg._cfg_dict.test_json_file]
        self.load_annotations()
        # self.h_samples = list(range(160, 720, 10))
        

    def load_annotations(self):
        self.logger.info('Loading LinkLabel annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                project: dict = json.load(anno_obj)

                for picture in project:
                    print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')
                    img_path = osp.join(self.data_root, picture["file_upload"])
                    current_img = cv2.imread(img_path)
                    for annotation in picture["annotations"]:
                        print(f'\t convert annotation with id {annotation["id"]}')
                        h_samples: list[int] = []
                        for result in annotation["result"]:
                            h_samples.append(result["value"]["y"])

                        h_samples.sort()
                        left: list[int] = [-2] * len(h_samples)
                        right: list[int] = [-2] * len(h_samples)
                        for result in annotation["result"]:
                            label = result["value"]["keypointlabels"]
                            if label == ["lane-left"]:
                                left[h_samples.index(result["value"]["y"])] = round(result["value"]["x"]*result["original_width"]/100)
                            elif label == ["lane-right"]:
                                right[h_samples.index(result["value"]["y"])] = round(result["value"]["x"]*result["original_width"]/100)
                            else:
                                print(f"Could not parse label: {label}")
                        
                        #create line for target_file
                        dic = {}
                        dic["lanes"] = [left, right]
                        dic["h_samples"] = h_samples
                        for i in range(len(h_samples)):
                            dic["h_samples"][i] = round(dic["h_samples"][i]*annotation["result"][0]["original_height"]/100) #Todo check for doubles
                        dic["raw_file"] = img_path

                        #recalculate to match tusimple y_samples
                        #dic = adjust_y_samples(dic)
                        with open(os.path.splitext(anno_file)[0] + '_converted.json', "a") as f:
                            j = json.dumps(dic)
                            f.write(j +'\n')
                    
                    lanes = [[(x, y) for (x, y) in zip(lane, h_samples) if x >= 0] for lane in dic["lanes"]]
                    lanes = [lane for lane in lanes if len(lane) > 0]
                    max_lanes = max(max_lanes, len(dic["lanes"]))
                    self.data_infos.append({
                        'img_path': img_path,
                        'img_name': picture["file_upload"],
                        # 'mask_path': osp.join(self.data_root, mask_path),# what's this?
                        'lanes': lanes, #Todo: Pair of x and y
                        'h_samples': h_samples,
                    })

            ##alt:           
            # for line in lines:
            #     data = json.loads(line)
            #     y_samples = data['h_samples']
            #     gt_lanes = data['lanes']
            #     mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
            #     lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
            #     lanes = [lane for lane in lanes if len(lane) > 0]
            #     max_lanes = max(max_lanes, len(lanes))
            #     self.data_infos.append({
            #         'img_path': osp.join(self.data_root, data['raw_file']),
            #         'img_name': data['raw_file'],
            #         'mask_path': osp.join(self.data_root, mask_path),
            #         'lanes': lanes,
            #     })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes
        
    def pred2lanes(self, pred, h_samples: list = list(range(160, 720, 10))):
        ys = np.array(h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred, self.data_infos[idx]['h_samples'])
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'linklabel_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes) # pass height samples from Label here
        result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        self.logger.info(result)
        return acc
