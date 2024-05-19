import os
import time
from datetime import datetime
import regex

class Log:
    def __init__(self, log_lines: list):
        self.log_lines: list[LogLine] = []
        for line in log_lines:
            if regex.search(LogLine.timestamp_pattern, line):
                self.log_lines.append(LogLine(line))

class LogLine:
    timestamp_pattern: str = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}'
    module_patttern: str = r'(?<=' + timestamp_pattern + r'\s-\s)(.*?)(?=\s-\s\w*\s\-\s.*)'
    level_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s)(\w*)(?=\s\-\s.*)'
    msg_pattern: str = r'(?<=' + timestamp_pattern + r'\s-\s.*?\s-\s\w*\s\-\s).*'

    def __init__(self, line: str):       
        self.timestamp = datetime.strptime(regex.search(LogLine.timestamp_pattern, line).group(), '%Y-%m-%d %H:%M:%S,%f')
        self.module: str = regex.search(LogLine.module_patttern, line).group()
        self.level: str = regex.search(LogLine.level_pattern, line).group()
        self.msg: str = regex.search(LogLine.msg_pattern, line).group()

        if self.msg.startswith("epoch:"):
            self = EpochLine(self.msg)
            print(self.epoch)
        
class EpochLine(LogLine):
    epoch_pattern: str = r'(?<=epoch:\s)(\d*)'

    def __init__(self, msg: str):
        # LogLine.__init__(self, line)
        self.epoch: int = int(regex.search(EpochLine.epoch_pattern, msg).group())
        self.step: int
        self.lr: float
        self.seg_loss: float
        self.exist_loss: float
        self.data: float
        self.batch: float
        self.eta: time #dureation


    

def parse_file(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        log: Log = Log(lines)

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__)) + "/example.log"
    parse_file(path)