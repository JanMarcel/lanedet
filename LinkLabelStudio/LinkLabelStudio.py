import json
import cv2

def convert(path: str):
    project: dict = readJSON(path)
    for picture in project:
        convert_pic(picture)

def readJSON(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def convert_pic(picture: dict):
    print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')   
    img = cv2.imread("LinkLabelStudio/" + picture["file_upload"])
    cv2.imshow('view', img)
    cv2.waitKey(0)
    for annotation in picture["annotations"]:
        convert_annotation(annotation)

def convert_annotation(annotation: dict):
    print(f'\t convert annotation with id {annotation["id"]}')
    for result in annotation["result"]:
        convert_annotation_result(result)

def convert_annotation_result(result: dict):
    print(f'\t\t convert result with id {result["id"]}')
    label = result["value"]["keypointlabels"]
    if label == ["lane-left"]:
        print("left")
    elif label == ["lane-right"]:
        print("right")
    else:
        print(f"Could not parse label: {label}")
    

    
    
convert('LinkLabelStudio/project-5-at-2024-11-20-07-55-53287f8c.json')
