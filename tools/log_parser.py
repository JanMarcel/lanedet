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
        times = []
        steps = []
        seg_losses = []
        exit_losses = []       
        metric_steps = []
        best_metrics = []
        tps = []
        fps = []
        fns = []
        precisions = []
        recalls = []
        f1s = []
        first: bool = True
        for line in self.log_lines:
            if isinstance(line, EpochLine):
                if first:
                    first = False
                    continue
                times.append(line.timestamp)
                steps.append(line.step)
                # seg_losses.append(line.seg_loss)
                # exit_losses.append(line.exist_loss)
                
            elif isinstance(line, JSONStatsLine):
                    metric_steps.append(steps[-1]) #applies for JSONStatsLine and BestMetricLine
                    tps.append(line.TP)
                    fps.append(line.FP)
                    fns.append(line.FN)
                    precisions.append(line.precision)
                    recalls.append(line.recall)
                    f1s.append(line.f1)
            elif isinstance(line, BestMetricLine):
                    best_metrics.append(line.best_metric)
    
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%HH'))
        #plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        # plt.plot(steps, seg_losses, label="seg_loss")
        # plt.plot(steps, exit_losses, label="exit_loss")
        plt.plot(metric_steps, best_metrics, label="best_metric")
        #Todo: following stats in extra graph --> dimensions too different
        plt.plot(metric_steps, tps, label="tp")
        plt.plot(metric_steps, fps, label="fp")
        plt.plot(metric_steps, fns, label="fn")
        plt.plot(metric_steps, precisions, label="precision")
        plt.plot(metric_steps, recalls, label="recall")
        plt.plot(metric_steps, f1s, label="f1")
        plt.legend(loc="upper right")
        #Todo: axes labels, title of graph
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