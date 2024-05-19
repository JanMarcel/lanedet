import cv2

filename = 'C:\\Projects\\AIT\\Teamprojekt\\driver_23_30frame\\driver_23_30frame\\05171114_0770.MP4\\00020.jpg'
img = cv2.imread(filename)
cv2.imshow('image', img)

with open(filename.removesuffix('.jpg') + '.lines.txt') as f:
    for line in f:
        data = line.split()
        
        for i in range(0, len(data), 2):
            x = float(data[i])
            y = float(data[i+1])

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
cv2.imshow('image', img)
cv2.waitKey(0)