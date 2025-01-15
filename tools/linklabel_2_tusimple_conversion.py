import json
import cv2
import os
import numpy as np
import argparse

global current_img

def convert(path: str, clip_path: str, show: bool = False):
    project: dict = readJSON(path)
    if not os.path.exists(os.path.splitext(path)[0] + '_converted.json'):
        for picture in project:
            convert_pic(picture, path, clip_path, show)
    else:
        print("Already converted -> delete/rename old file to convert again")

def readJSON(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def convert_pic(picture: dict, path: str, clip_path: str, show: bool):
    global current_img
    print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')
    if show:
        try:
            current_img = cv2.imread(clip_path + picture["file_upload"])
        except Exception as e:
            print(f"Could not read image {clip_path + picture['file_upload']}")
            print(e)
            current_img = None
            show = False
    for annotation in picture["annotations"]:
        convert_annotation(annotation, path, (clip_path + picture["file_upload"]).replace("data/LinkLabel/", ""), show)
    if show:
        cv2.imshow(picture["file_upload"], current_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def convert_annotation(annotation: dict, path: str, pic_path: str, show: bool):
    print(f'\t convert annotation with id {annotation["id"]}')
    lanes = {}
    for result in annotation["result"]:
        label = result["value"]["keypointlabels"]
        if len(label) > 1:
            raise Exception("Label has more than one element")
        label = label[0]
        
        if label not in lanes:
            lanes[label] = []
        #collect all points sorted by label
        lanes[label].append((round(result["value"]["x"]*result["original_width"]/100), round(result["value"]["y"]*result["original_height"]/100)))
        if show:
            cv2.circle(current_img, lanes[label][-1], radius=5, color=(0, 0, 255), thickness=-1)

    adjusted_lanes, h_samples = adjust_y_samples(lanes)
    #create line for target_file
    dic = {}
    dic["lanes"] = adjusted_lanes
    dic["h_samples"] = h_samples
    dic["raw_file"] = pic_path
    with open(os.path.splitext(path)[0] + '_converted.json', "a") as f:
        j = json.dumps(dic)
        f.write(j +'\n')
    

def adjust_y_samples(lanes: list, y_samples: list[int]=list(range(160, 720, 10))) -> dict:
    global current_img
    adjusted_lanes = []
    for l_name, lane in lanes.items():
        orig_y = [point[1] for point in lane]
        valid_lane = [(point[1], point[0]) for point in lane if point[0] != -2] #exchange x and y for easier x calculation
        if len(valid_lane) > 2:
            pol = numpy_polyfit(valid_lane, 3)
            x_lane: list[int] = [-2] * len(y_samples)
            for y in y_samples:
                if y > min(orig_y) and y < max(orig_y): # dont predict line
                    x = int(pol(y))
                    if x >= 0: # dont predict line outside of image
                        x_lane[y_samples.index(y)] = x
                        if current_img is not None:
                            cv2.circle(current_img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            adjusted_lanes.append(x_lane)
    return adjusted_lanes, y_samples


def numpy_polyfit(points, degree=None):
    x_vals, y_vals = zip(*points)  # Separate x and y values
    if degree is None:
        degree = len(points) - 1  # Degree is n-1 for n points

    coefficients = np.polyfit(x_vals, y_vals, degree)
    polynomial = np.poly1d(coefficients)
    return polynomial
    
#convert('data/LinkLabel/project-5-at-2024-12-04-05-55-e9304360.json', 'data/LinkLabel/clips/241203/', True)
# example call assuming you are in the root directory of the project:
# python .\tools\linklabel_2_tusimple_conversion.py --show True 'data/LinkLabel/project-5-at-2024-12-04-05-55-e9304360.json' 'data/LinkLabel/clips/241203/'              
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LinkLabel to TuSimple format')
    parser.add_argument('path', type=str, help='Path to LinkLabel JSON')
    parser.add_argument('clip_path', type=str, help='Path directory of clips')
    parser.add_argument('--show', type=bool, help='Show images')
    args = parser.parse_args()
    convert(args.path, args.clip_path, args.show)