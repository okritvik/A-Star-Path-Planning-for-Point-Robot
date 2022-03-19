import cv2
import numpy as np
 
 
# Reading an image in default mode
image = np.zeros((200,200,3),np.uint8)
 
# Window name in which image is displayed
window_name = 'Image'
 
# Start coordinate, here (225, 0)
# represents the top right corner of image
start_point = (50, 0)
 
# End coordinate
end_point = (0, 90.5)
 
# Red color in BGR
color = (0, 0, 255)
 
# Line thickness of 9 px
thickness = 5
 
# Using cv2.arrowedLine() method
# Draw a red arrow line
# with thickness of 9 px and tipLength = 0.5
image = cv2.arrowedLine(image, start_point, end_point,
                    color, thickness, tipLength = 0.5)
 
# Displaying the image
cv2.imshow(window_name, image)