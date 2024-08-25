import torch
from lanedet.models.backbones.mobilenet import MobileNetV2
import cv2
import numpy as np
import pickle
import argparse

from lanedet.models.registry import build_net
from lanedet.utils.config import Config

def torch_to_onnx(model, input_tensor, onnx_path):
    model.eval()
    model = model.to(input_tensor.device)
    with torch.no_grad():
        torch.onnx.export(model, input_tensor, onnx_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)  # You might need to adjust the opset_version

def main(cfg):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net(cfg)
    # model = MobileNetV2()
    # model = torch.nn.parallel.DataParallel(
    #             model, device_ids = range(1)).cuda()
    # model.load_state_dict(
    #     torch.load('models/mobilenet_tusimple_200epochs.pth', map_location="cuda"), strict=True)
    model = model.to("cuda:0")  # Ensure the model is on the correct device
    model = model.eval()
    # random_number_moves = random.randint(28, 2 * board_size * board_size)
    # args = {
    #     'C': 2,
    #     'num_searches': 100,
    #     'dirichlet_epsilon': 0,
    #     'dirichlet_alpha': 0.1
    # }
    # mcts = MCTS(game, args, model)
    # player = -1

    # for _ in range(random_number_moves):
    #     neutral_state = game.change_perspective(
    #         game.state.copy(), player)
    #     mcts_probs = mcts.search(neutral_state)
    #     action = numpy.argmax(mcts_probs)
    #     state = game.step(game.state, action, player)
    #     player = -player
    #     yield state

    # encoded_state = game.get_encoded_state(state)
    # tensor_state = torch.tensor(
    #     encoded_state, 
    #     dtype=torch.float32).unsqueeze(0).to(device)

    # input_img = torch.randn(1, 3, 360, 640).cuda()
    with open('img2.pkl', 'rb') as fptr:
        input_img = pickle.loads(fptr.read())
    
    tensor_state = {'img':torch.from_numpy(input_img).cuda()}

    #tensor_state = torch.randn(3, 640, 360)

    # Convert PyTorch model to ONNX
    onnx_path = 'models/mobilenetv2_model_200epochs.onnx'
    torch.onnx.dynamo_export(model=model, args=tensor_state, f=onnx_path, export_params=True, opset_version=14, verbose=True)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    # process(args)
    cfg = Config.fromfile(args.config)
    main(cfg)