import json
import cv2
import os

global current_img

def convert(path: str, clip_path: str):
    project: dict = readJSON(path)
    if not os.path.exists(os.path.splitext(path)[0] + '_converted.json'):
        for picture in project:
            convert_pic(picture, path, clip_path)
    else:
        print("Already converted")

def readJSON(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def convert_pic(picture: dict, path: str, clip_path: str):
    global current_img
    print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')
    current_img = cv2.imread(clip_path + picture["file_upload"])
    for annotation in picture["annotations"]: 
        convert_annotation(annotation)
    cv2.imshow('view', current_img)
    cv2.waitKey(0)

def convert_annotation(annotation: dict, path: str, pic_path: str):
    print(f'\t convert annotation with id {annotation["id"]}')
    h_samples: list[int] = []
    for result in annotation["result"]:
        h_samples.append(result["value"]["y"])
        #convert_annotation_result(result)
    h_samples.sort()
    left: list[int] = [-2] * len(h_samples)
    right: list[int] = [-2] * len(h_samples)
    for result in annotation["result"]:
        label = result["value"]["keypointlabels"]
        if label == ["lane-left"]:
            left[h_samples.index(result["value"]["y"])] = round(result["value"]["x"]*result["original_width"]/100)
        elif label == ["lane-right"]:
            right[h_samples.index(result["value"]["y"])] = round(result["value"]["x"]*result["original_width"]/100)
        else:
            print(f"Could not parse label: {label}")
    
    #create line for target_file
    dic = {}
    dic["lanes"] = [left, right]
    dic["h_samples"] = h_samples
    for i in range(len(h_samples)):
        dic["h_samples"][i] = round(dic["h_samples"][i]*annotation["result"][0]["original_height"]/100) #Todo check for doubles
    dic["raw_file"] = pic_path

    print(dic)
    with open(os.path.splitext(path)[0] + '_converted.json', "a") as f:
        j = json.dumps(dic)
        f.write(j +'\n')

def convert_annotation_result(result: dict):
    global current_img
    print(f'\t\t convert result with id {result["id"]}')
    label = result["value"]["keypointlabels"]

    if label == ["lane-left"]:
        print("left")
    elif label == ["lane-right"]:
        print("right")
    else:
        print(f"Could not parse label: {label}")
    
    point = (round(result["value"]["x"]/100 * result["original_width"]), round(result["value"]["y"]/100 * result["original_height"]))
    print(point)
    current_img = cv2.circle(current_img, point, radius=5, color=(0, 0, 255), thickness=-1)    

    
convert('data/LinkLabel/project-5-at-2024-12-04-05-55-e9304360.json', 'data/LinkLabel/clips/241203/')
