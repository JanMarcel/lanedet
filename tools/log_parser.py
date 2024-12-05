import os
import time
from datetime import datetime
import json
import regex
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Log:
    head_pattern: str = r'(?<=\(heads\):\s)(.*?)(?=\()'
    backbone_pattern: str = r'(?<=\(backbone\):\s)(.*?)(?=\()'
    dataset_pattern: str = r"(?<=dataset_type\s=\s')(.*?)(?=')"

    def __init__(self, log_lines: list):
        self.log_lines: list[LogLine] = []
        self.head: str = None
        for line in log_lines:
            if regex.search(LogLine.timestamp_pattern, line):
                self.log_lines.append(LogLine.from_line(line, self.head))
            elif regex.search(Log.head_pattern, line):
                self.head: str = regex.search(Log.head_pattern, line).group()
            elif regex.search(Log.backbone_pattern, line):
                self.backbone: str = regex.search(Log.backbone_pattern, line).group()
            elif regex.search(Log.dataset_pattern, line): # Todo nicht alle configs gleich!
                self.dataset: str = regex.search(Log.dataset_pattern, line).group()

    def plot(self):
        #Todo: 
        # 2 graphs one for numbers between 0 and 1, one for > 1
        # depending on head, different metrics

        __laneatt = ["loss", "cls_loss", "reg_loss"]

        if self.head == "LaneATT":
            step_keys = __laneatt
        #Todo: other heads
        
        epoch_keys_small = ["best_metric", "precision", "recall", "f1"]
        epoch_keys_big = ["TP", "FP", "FN"]

        plot_dict = {}
        plot_dict["epochs"] = [0]
        plot_dict["steps"] = []
        plot_dict["step_values"] = {}
        plot_dict["epoch_values_small"] = {}
        plot_dict["epoch_values_big"] = {}

        for k in step_keys:
            plot_dict["step_values"][k] = []
        
        for k in epoch_keys_small:
            plot_dict["epoch_values_small"][k] = []

        for k in epoch_keys_big:
            plot_dict["epoch_values_big"][k] = []

        for line in self.log_lines:
            if isinstance(line, EpochLine):
                if plot_dict["epochs"][-1] != line.epoch:
                    plot_dict["epochs"].append(line.epoch)
                plot_dict["steps"].append(line.step)
                for key in step_keys:
                    plot_dict["step_values"][key].append(getattr(line, key))
            if isinstance(line, BestMetricLine):
                plot_dict["epoch_values_small"]["best_metric"].append(line.best_metric)
            if isinstance(line, JSONStatsLine):
                for key in epoch_keys_small:
                    if key != "best_metric":
                        plot_dict["epoch_values_small"][key].append(getattr(line, key))
                for key in epoch_keys_big:
                        plot_dict["epoch_values_big"][key].append(getattr(line, key))
                

        f1 = plt.figure("1")
        for key in step_keys:
            plt.plot(plot_dict["steps"], plot_dict["step_values"][key], label=key)
            plt.legend()
        ax = plt.gca()
        #ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, 10])
        f2 = plt.figure("2")
        for key in plot_dict["epoch_values_small"]:
            plt.plot(plot_dict["epochs"], plot_dict["epoch_values_small"][key], label=key)
            plt.legend()

        f3 = plt.figure("3")
        for key in plot_dict["epoch_values_big"]:
            plt.plot(plot_dict["epochs"], plot_dict["epoch_values_big"][key], label=key)
            plt.legend()
        
        f1.show()
        f2.show()
        f3.show()
        plt.waitforbuttonpress()              
class LogLine:
    timestamp_pattern: str = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}'
    module_patttern: str = r'(?<=' + timestamp_pattern + r'\s-\s)(.*?)(?=\s-\s\w*\s\-\s.*)'
    level_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s)(\w*)(?=\s\-\s.*)'
    msg_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s\w*\s\-\s).*'

    @staticmethod
    def from_line(line: str, head: str):
        timestamp = datetime.strptime(regex.search(LogLine.timestamp_pattern, line).group(), '%Y-%m-%d %H:%M:%S,%f')
        module: str = regex.search(LogLine.module_patttern, line).group()
        level: str = regex.search(LogLine.level_pattern, line).group()
        msg: str = regex.search(LogLine.msg_pattern, line).group()

        logline: LogLine = LogLine(timestamp, module, level, msg)
        if module == "lanedet.utils.recorder":
            if msg.startswith("epoch:"):
                return EpochLine(logline, head)

            elif msg.startswith("Best metric:"):
                return BestMetricLine(logline)
        elif module == "lanedet.datasets.base_dataset":
            try:
                stats: dict = json.loads(msg)
                return JSONStatsLine(logline, stats)
            except ValueError as e:
                print(e)
        
        # fallback
        return logline

    def __init__(self, timestamp: datetime, module: str, level: str, msg: str):       
        self.timestamp = timestamp
        self.module: str = module
        self.level: str = level
        self.msg: str = msg
        
class EpochLine(LogLine):
    #shared patterns
    epoch_pattern: str = r'(?<=epoch:\s)(\d*)'
    step_pattern: str = r'(?<=step:\s)(\d*)'
    lr_pattern: str = r'(?<=lr:\s)(\d*\.\d*)'
    data_pattern: str = r'(?<=data:\s)(\d*\.\d*)'
    batch_pattern: str = r'(?<=batch:\s)(\d*\.\d*)'
    eta_pattern: str = r'(?<=ETA:\s)(\d*\.\d*)'
    #head specific patterns
    #LaneATT
    loss_pattern: str = r'(?<=loss:\s)(\d*\.\d*)'
    clc_loss_pattern: str = r'(?<=clc_loss:\s)(\d*\.\d*)'
    reg_loss_pattern: str = r'(?<=reg_loss:\s)(\d*\.\d*)'
    #CondLaneHead
    hm_loss_pattern: str = r'(?<=hm_loss:\s)(\d*\.\d*)'
    kps_loss_pattern: str = r'(?<=kps_loss:\s)(\d*\.\d*)'
    row_loss_pattern: str = r'(?<=row_loss:\s)(\d*\.\d*)'
    range_loss_pattern: str = r'(?<=range_loss:\s)(\d*\.\d*)'
    #LaneSeg
    seg_loss_pattern: str = r'(?<=seg_loss:\s)(\d*\.\d*)'
    #LaneCls
    cls_loss_pattern: str = r'(?<=cls_loss:\s)(\d*\.\d*)'

    #exampleLog
    #exist_loss_pattern: str = r'(?<=exist_loss:\s)(\d*\.\d*)'


    def __init__(self, logline: LogLine, head: str):
        super().__init__(logline.timestamp, logline.module, logline.level, logline.msg)
        #shared properties
        self.epoch: int = int(regex.search(EpochLine.epoch_pattern, logline.msg).group())
        self.step: int = int(regex.search(EpochLine.step_pattern, logline.msg).group())
        self.lr: float = float(regex.search(EpochLine.lr_pattern, logline.msg).group())
        self.data: float = float(regex.search(EpochLine.data_pattern, logline.msg).group())
        self.batch: float = float(regex.search(EpochLine.batch_pattern, logline.msg).group())
        self.eta: time

        #head specific properties
        if head == "LaneATT":
            self.loss: float = float(regex.search(EpochLine.loss_pattern, logline.msg).group())
            self.cls_loss: float = float(regex.search(EpochLine.cls_loss_pattern, logline.msg).group())
            self.reg_loss: float = float(regex.search(EpochLine.reg_loss_pattern, logline.msg).group())
        elif head == "CondLaneHead":
            self.loss: float = float(regex.search(EpochLine.loss_pattern, logline.msg).group())
            self.hm_loss: float = float(regex.search(EpochLine.hm_loss_pattern, logline.msg).group())
            self.kps_loss: float = float(regex.search(EpochLine.kps_loss_pattern, logline.msg).group())
            self.row_loss: float = float(regex.search(EpochLine.row_loss_pattern, logline.msg).group())
            self.range_loss: float = float(regex.search(EpochLine.range_loss_pattern, logline.msg).group())
        elif head == "LaneSeg":
            self.seg_loss: float = float(regex.search(EpochLine.seg_loss_pattern, logline.msg).group())
        elif head == "LaneCls":
            self.cls_loss: float = float(regex.search(EpochLine.cls_loss_pattern, logline.msg).group())
        else:
            print("Unknown head: " + head)
        #self.exist_loss: float = float(regex.search(EpochLine.exist_loss_pattern, logline.msg).group())


class BestMetricLine(LogLine):
    best_metric_pattern: str = r'(?<=Best metric:\s)(\d.\d*)'

    def __init__(self, logline: LogLine):
        super().__init__(logline.timestamp, logline.module, logline.level, logline.msg)
        self.best_metric: float = float(regex.search(BestMetricLine.best_metric_pattern, logline.msg).group())

class JSONStatsLine(LogLine):
    def __init__(self, logline: LogLine, stats: dict):
        super().__init__(logline.timestamp, logline.module, logline.level, logline.msg)
        self.TP = stats['TP']
        self.FP = stats['FP']
        self.FN = stats['FN']
        self.precision = stats['Precision']
        self.recall = stats['Recall']
        self.f1 = stats['F1']

def parse_file(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        log: Log = Log(lines)
        log.plot()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Parse log file and plot metrics.')
    # parser.add_argument('log_file', type=str, help='Path to the log file')
    # args = parser.parse_args()
    # path = args.log_file + "/log.txt"
    path = ".\\work_dirs\\CULane\\20241126_032949_lr_3e-04_b_8\\" + "/log.txt"
    #path = os.path.dirname(os.path.abspath(__file__)) + "/example.log"

    parse_file(path)