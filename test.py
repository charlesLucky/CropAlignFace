import cv2
from mtcnn import MTCNN

detector = MTCNN()

image = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
imgs = [image,image2,image3]
print(imgs)
result = detector.detect_faces(imgs)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
bounding_box = result[0]['box'] #The bounding box is formatted as [x, y, width, height] under the key 'box'.
keypoints = result[0]['keypoints']

print(bounding_box)
crop_img = image[int(bounding_box[1]):int(bounding_box[1])+int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[0])+int(bounding_box[2])]

cv2.imwrite("ivan_crop_img.jpg", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

print(result)



