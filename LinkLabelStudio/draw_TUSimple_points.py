import json
import cv2
import os

def show_labels(path: str):
    with open(path, 'r') as file:
        for line in file:
            labels: dict = json.loads(line)
            print(os.path.dirname(path))
            img = cv2.imread(os.path.dirname(path) + "/" + labels["raw_file"])

            for lane in labels["lanes"]:
                for i in range(len(lane) - 1):
                    if lane[i] != -2:
                        cv2.circle(img, (lane[i], labels["h_samples"][i]), radius=5, color=(0, 0, 255), thickness=-1)    

            cv2.imshow('view', img)
            cv2.waitKey(0)



# show_labels("data\TUSimple\label_data_0313.json")

show_labels("LinkLabelStudio\\target_file.json")