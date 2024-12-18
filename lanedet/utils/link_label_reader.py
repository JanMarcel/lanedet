import numpy as np
import json
import os.path as osp
import cv2
import os

def load_link_label_project(path: str, data_root: str):
    data_infos = []
    max_lanes = 0
    with open(path, 'r') as anno_obj:
        project: dict = json.load(anno_obj)

        for picture in project:
            print(f'convert picture with id {picture["id"]} and name {picture["file_upload"]}')
            img_path = osp.join(data_root, picture["file_upload"])
            current_img = cv2.imread(img_path)
            for annotation in picture["annotations"]:
                print(f'\t convert annotation with id {annotation["id"]}')
                h_samples: list[int] = []
                for result in annotation["result"]:
                    h_samples.append(result["value"]["y"])

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
                dic["raw_file"] = img_path

                #recalculate to match tusimple y_samples
                #dic = adjust_y_samples(dic)
                with open(os.path.splitext(path)[0] + '_converted.json', "a") as f:
                    j = json.dumps(dic)
                    f.write(j +'\n')
                
                lanes = [[(x, y) for (x, y) in zip(lane, h_samples) if x >= 0] for lane in dic["lanes"]]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(dic["lanes"]))
                data_infos.append({
                    'img_path': img_path,
                    'img_name': picture["file_upload"],
                    # 'mask_path': osp.join(self.data_root, mask_path),# what's this?
                    'lanes': lanes, #Todo: Pair of x and y
                    'h_samples': h_samples,
                    'raw_file': picture["file_upload"],
                })

    return (data_infos, max_lanes)

# def adjust_y_samples(dic: dict, y_samples: list[int]=list(range(160, 720, 10))) -> dict:
    # lanes = []
    # ret = {}
    # ret["lanes"] = []
    # ret["h_samples"] = y_samples
    # ret["raw_file"] = dic["raw_file"]
    # for lane in dic["lanes"]:
    #     lane_points = []
    #     for i in range(len(lane) - 1):
    #         if lane[i] != -2:
    #             lane_points.append((dic["h_samples"][i], lane[i]))
    #     lanes.append(lane_points)
    
    # for lane in lanes:
    #     print(lane)
    #     x_lane: list[int] = [-2] * len(y_samples)
    #     if len(lane) > 2:
    #         pol = numpy_polyfit(lane)        
    #         for y in y_samples:
    #             if y > min(dic["h_samples"]) and y < max(dic["h_samples"]): # dont predict line
    #                 x = int(pol(y))
    #                 if x >= 0: # dont predict line outside of image
    #                     x_lane[y_samples.index(y)] = x
    #     ret["lanes"].append(x_lane)
    
    # return ret

# def numpy_polyfit(points, degree=None):
    # x_vals, y_vals = zip(*points)  # Separate x and y values
    # if degree is None:
    #     degree = len(points) - 1  # Degree is n-1 for n points

    # coefficients = np.polyfit(x_vals, y_vals, degree)
    # polynomial = np.poly1d(coefficients)
    # return polynomial