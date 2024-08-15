import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import onnx
import cv2
import numpy as np
from lanedet.models.backbones.mobilenet import MobileNetV2
from lanedet.models.nets.detector import Detector, to_json
from lanedet.utils.config import Config
from lanedet.datasets.process.transforms import to_tensor
from detect import Detect, get_img_paths
from tqdm import tqdm
import argparse
import json

def torch_to_onnx(model, input_tensor, onnx_path):
    model.eval()
    model = model.to(input_tensor.device)
    with torch.no_grad():
        torch.onnx.export(model, input_tensor, onnx_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)  # You might need to adjust the opset_version

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = MobileNetV2()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    model = Detector(cfg)
    #model.load_state_dict(
        #torch.load('models/mobilenet_tusimple_200epochs.pth', map_location="cuda"), strict=False)

    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        data = detect.preprocess(p)
        #data['lanes'] = detect.inference(data)[0]
        #detect.run(p)
    
    #tensor_state = torch.randn(1, 3, 640, 360)
    #tensor_state = torch.randn(3, 640, 360)
    #img_path = '/home/ti121alk/Dokumente/lanedet/data/Labor/racecar_image_1.jpg'
    #ori_img = cv2.imread(img_path)
    ##img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    #img = torch.load('batch_img.pth')
    #batch = { # is now exactly the same as the batch when called from detect.py
    #    'img': to_tensor(img), # tensor
    #    'img_path': img_path,
    #    'ori_img': ori_img,#array
#
#    }

    
    #with open('batch2.json', 'w') as fp:
    #    json.dump(to_json(batch), fp, indent=4, sort_keys=True)

    #batch = torch.randn(1, 1280, 12, 20)
    #batch = torch.randn(1, 3, 640, 360)
    # Convert PyTorch model to ONNX
    onnx_path = 'models/mobilenetv2_model_200epochs.onnx'
    #try:
    #batch = to_tensor(img)
    #print(f"type of data: {data}")
    torch.cuda.synchronize()
    torch.onnx.export(detect.net.module, data['img'].cuda(), onnx_path, opset_version=9, verbose=True)
    #except Exception as e:
    #    print(e)
    # torch_to_onnx(model, tensor_state, onnx_path)
    print(f"ONNX model saved to {onnx_path}")

    # # Convert ONNX model to TensorFlow
    # # Load ONNX model
    # onnx_model = onnx.load(onnx_path)
    # tf_rep = onnx_tf.backend.prepare(onnx_model)
    # tf_rep.export_graph('resnet_model.pb')
    # print("TensorFlow model saved to resnet_model.pb")

    # # Convert TensorFlow model to TFLite
    # # Load the TensorFlow model
    # converter = tf.lite.TFLiteConverter.from_saved_model('resnet_model.pb')
    # tflite_model = converter.convert()

    # # Save the TFLite model
    # with open('resnet_model.tflite', 'wb') as f:
    #     f.write(tflite_model)


if __name__ == "__main__":
    main()