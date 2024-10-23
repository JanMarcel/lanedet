import pyrealsense2 as rs
import numpy as np
import cv2
from time import sleep

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

counter = 108
print("waiting 10s to init camera")
sleep(10)
try:
    while True:
        sleep(1)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if color_frame:            
            cv_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(f"./lab_pics/test_{counter}.jpg", cv_image)
            counter += 1
            print(f"Successfully saved pic #{counter}!")
except KeyboardInterrupt:
    pass

# frames = pipeline.wait_for_frames()
# counter = 1
# #while True:
# color_frame = frames.get_color_frame()

# if color_frame:            
#     cv_image = np.asanyarray(color_frame.get_data())
#     cv2.imwrite(f"test_{counter}.jpg", cv_image)
#     counter += 1



# #except KeyboardInterrupt:
# #    pass