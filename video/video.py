import cv2
vidcap = cv2.VideoCapture('video/2024-02-03-18-51-57.mp4')
success,image = vidcap.read()
count = 1
while success:
  cv2.imwrite("video/seg3/frame%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1