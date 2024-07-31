import torch
import onnx
from lanedet.models.backbones.mobilenet import MobileNetV2

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
    model = MobileNetV2()
    model.load_state_dict(
        torch.load('models/mobilenet_tusimple_200epochs.pth', map_location="cuda"), strict=False)
    # model = model.to(device)  # Ensure the model is on the correct device
    # model.eval()
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

    tensor_state = torch.randn(1, 3, 640, 360)
    #tensor_state = torch.randn(3, 640, 360)

    # Convert PyTorch model to ONNX
    onnx_path = 'models/mobilenetv2_model_200epochs.onnx'
    torch.onnx.export(model, tensor_state, onnx_path)
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