import json
import cv2
global current_img

def convert(path: str):
    project: dict = readJSON(path)
    for picture in project:
        convert_pic(picture)

def readJSON(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def convert_pic(picture: dict):
    global current_img
    print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')   
    current_img = cv2.imread("LinkLabelStudio/" + picture["file_upload"])
    for annotation in picture["annotations"]: 
        convert_annotation(annotation)
    cv2.imshow('view', current_img)
    cv2.waitKey(0)

def convert_annotation(annotation: dict):
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
    dic["raw_file"] = "d3b4989f-test_0.jpg" #+ annotation["file_upload"] #Todo think about directory strucure

    print(dic)
    with open("LinkLabelStudio/target_file.json", "w") as f:
        json.dump(dic, f)

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

    
convert('LinkLabelStudio/project-5-at-2024-11-20-07-55-53287f8c.json')
