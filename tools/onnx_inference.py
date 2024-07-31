import onnxruntime as ort
import cv2
import numpy as np
import torch
from torch import nn

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
    #proposals_list = output['proposals_list']
    proposals_list = output
    softmax = nn.Softmax(dim=1)
    decoded = []
    for proposals, _, _, _ in proposals_list:
        proposals[:, :2] = softmax(proposals[:, :2])
        proposals[:, 4] = torch.round(proposals[:, 4])
        if proposals.shape[0] == 0:
            decoded.append([])
            continue
        if as_lanes:
            pred = self.proposals_to_pred(proposals)
        else:
            pred = proposals
        decoded.append(pred)

    return decoded


cvimg = cv2.imread("data/Labor/racecar_image_3.jpg")
down_points = (640, 360)

cvimg = cv2.resize(cvimg, down_points, interpolation= cv2.INTER_LINEAR)
input_img = np.asanyarray(cvimg, dtype=np.float32).reshape(1, 3, 640, 360)

def to_numpy(tensor):
    return tensor.cpu().numpy()

ort_session = ort.InferenceSession('models/mobilenetv2_model_200epochs.onnx')
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: input_img}
outputs = ort_session.run(
    None,
    ort_inputs
)
print(outputs)
print(outputs[0].shape)
print(get_lanes(outputs))
#combo_image = cv2.addWeighted(cvimg, 0.8, outputs[0], 1, 1)
#cv2.imshow(combo_image)


print(outputs)