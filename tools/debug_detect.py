import os
os.system("python tools/detect.py configs/laneatt/mobilenetv2_200_epochs.py --img data/Labor/ --load_from models/mobilenet_tusimple_200_epochs.pth")