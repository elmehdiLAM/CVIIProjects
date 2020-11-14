import numpy as np
import matplotlib.pyplot as plt
import cv2
from classifier import classifier

img=np.zeros((80,80,3),"uint8")
# show image
cv2.imshow('image', img)

def drawpoint(img, x, y):
    img[y, x] = 255
    img[y - 1, x - 1] = 255
    img[y - 1, x] = 255
    img[y, x - 1] = 255
    img[y - 1, x + 1] = 255
    img[y + 1, x - 1] = 255
    img[y + 1, x + 1] = 255
    img[y, x + 1] = 255
    img[y + 1, x] = 255


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_LBUTTON:
            print(f"coordonnées {x} and {y}")
            drawpoint(img, x, y)
            cv2.imshow('image', img)


cv2.setMouseCallback('image', mouse_click)
cv2.waitKey(0)
image=cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
# close all the opened windows
plt.imshow(image)
plt.show()
cv2.destroyAllWindows()
# let predict our numbers
number=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
print(number.shape)

print(classifier.predict(number.reshape(1,64)))

