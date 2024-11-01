import os
import time
from datetime import datetime
import regex
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Log:
    def __init__(self, log_lines: list):
        self.log_lines: list[LogLine] = []
        for line in log_lines:
            if regex.search(LogLine.timestamp_pattern, line):
                self.log_lines.append(LogLine.from_line(line))
    
    def plot(self):
        x = []
        y = []
        y1 = []

        first: bool = True
        for line in self.log_lines:
            if isinstance(line, EpochLine):
                if first:
                    first = False
                    continue
                x.append(line.timestamp)
                y.append(line.seg_loss)
                y1.append(line.exist_loss)
    
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%HH'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.plot(x, y)
        plt.plot(x, y1)
        plt.waitforbuttonpress()

class LogLine:
    timestamp_pattern: str = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}'
    module_patttern: str = r'(?<=' + timestamp_pattern + r'\s-\s)(.*?)(?=\s-\s\w*\s\-\s.*)'
    level_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s)(\w*)(?=\s\-\s.*)'
    msg_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s\w*\s\-\s).*'

    @staticmethod
    def from_line(line: str):
        timestamp = datetime.strptime(regex.search(LogLine.timestamp_pattern, line).group(), '%Y-%m-%d %H:%M:%S,%f')
        module: str = regex.search(LogLine.module_patttern, line).group()
        level: str = regex.search(LogLine.level_pattern, line).group()
        msg: str = regex.search(LogLine.msg_pattern, line).group()

        logline: LogLine = LogLine(timestamp, module, level, msg)
        if msg.startswith("epoch:"):
            return EpochLine(logline)
        else:
            return logline

    def __init__(self, timestamp: datetime, module: str, level: str, msg: str):       
        self.timestamp = timestamp
        self.module: str = module
        self.level: str = level
        self.msg: str = msg
        
class EpochLine(LogLine):
    epoch_pattern: str = r'(?<=epoch:\s)(\d*)'
    step_pattern: str = r'(?<=step:\s)(\d*)'
    lr_pattern: str = r'(?<=lr:\s)(\d*\.\d*)'
    seg_loss_pattern: str = r'(?<=seg_loss:\s)(\d*\.\d*)'
    exist_loss_pattern: str = r'(?<=exist_loss:\s)(\d*\.\d*)'
    data_pattern: str = r'(?<=data:\s)(\d*\.\d*)'
    batch_pattern: str = r'(?<=batch:\s)(\d*\.\d*)'
    eta_pattern: str = r'(?<=ETA:\s)(\d*\.\d*)'

    def __init__(self, logline: LogLine):
        super().__init__(logline.timestamp, logline.module, logline.level, logline.msg)
        self.epoch: int = int(regex.search(EpochLine.epoch_pattern, logline.msg).group())
        self.step: int = int(regex.search(EpochLine.step_pattern, logline.msg).group())
        self.lr: float = float(regex.search(EpochLine.lr_pattern, logline.msg).group())
        self.seg_loss: float = float(regex.search(EpochLine.seg_loss_pattern, logline.msg).group())
        self.exist_loss: float = float(regex.search(EpochLine.exist_loss_pattern, logline.msg).group())
        self.data: float = float(regex.search(EpochLine.data_pattern, logline.msg).group())
        self.batch: float = float(regex.search(EpochLine.batch_pattern, logline.msg).group())
        #self.eta: time 
        print(self)


    

def parse_file(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        log: Log = Log(lines)
        log.plot()

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__)) + "/example.log"
    parse_file(path)