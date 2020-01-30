import numpy as np
from scipy import signal
import cv2
import sys
import time
import seaborn as sns
import matplotlib.pyplot as plt

TOP_LEFT = (270, 230) 
BOTTOM_RIGHT = (388, 438)
dataset = []

if len(sys.argv) <= 1:
    raise ValueError("You need to specify the video file")

def autocrop(image, threshold=13):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.
    Credit: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        return [cols[0], cols[-1] + 1, rows[0], rows[-1] + 1]


count = 0
capture = cv2.VideoCapture(sys.argv[1])
for _ in range(200):
    (grabbed, frame) = capture.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dims = autocrop(grayframe)
    if count == 0:
        smallest_dims = dims
    else:
        smallest_dims = [max(dims[0], smallest_dims[0]), max(dims[1], smallest_dims[1]), max(dims[2], smallest_dims[2]), max(dims[3], smallest_dims[3])]

capture = cv2.VideoCapture(sys.argv[1])
while True:
    (grabbed, frame) = capture.read()
    if not grabbed:
        break

    # frame = frame[smallest_dims[0]:smallest_dims[1], smallest_dims[2]:smallest_dims[3]]
    frame = cv2.resize(frame, (600, 600))

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Credit: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    # alpha = 5 # Contrast control (1.0-3.0)
    # beta = -50 # Brightness control (0-100)
    # alpha = 4 # Contrast control (1.0-3.0)
    # beta = -50 # Brightness control (0-100)
    # contrastframe = cv2.convertScaleAbs(grayframe, alpha=alpha, beta=beta)
    contrastframe = grayframe

    cropped = contrastframe[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]] 
    # cropped = contrastframe 
    cv2.imshow('Video', cropped) 
    dataset.append(cv2.mean(cropped)[0])

    keyVal = cv2.waitKey(1) & 0xFF

capture.release()
cv2.destroyAllWindows()

# analysis
x = [x for x in range(len(dataset))]
y = dataset
x_array = np.array(x)
print(x, y)
plot = sns.lineplot(x=x, y=y)
plt.show(plot)

np.save("6_dataset.pickle", y)
print(signal.find_peaks(y))
