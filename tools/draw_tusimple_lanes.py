import cv2
import json
import os

os.chdir(os.path.expanduser('~')+"/Dokumente/lanedet")
with open("data/tusimple/label_data_0313.json") as fptr:
    for line in fptr:
        first_row = line
        break
    data = json.loads(first_row)



img = cv2.imread(os.path.join("data/tusimple", data["raw_file"]))
h_samples = data["h_samples"]
for i in range(len(h_samples)):
    for lane in data["lanes"]:
        x,y = lane[i], h_samples[i]
        if x > 0:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

cv2.imshow("test", img)
cv2.waitKey(0)