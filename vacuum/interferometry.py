import numpy as np
import json
from scipy import signal
import cv2
import sys
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt

with open("trials.json", "r") as f:
    data = json.load(f)

dataset = []

for each_trial in data.keys():
    try:
        TOP = data[each_trial]["coord"]["top"]
        BOT = data[each_trial]["coord"]["bot"]
        TRIAL = each_trial 
        TEST = data[each_trial]["test"] 

        trial = []

        if data[each_trial]["skip"]:
            continue

        print(TOP, BOT)


        print("Trial:", TRIAL)

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
        capture = cv2.VideoCapture("real_data/" + str(TRIAL) + "_vac_data.mp4")
        for _ in range(200):
            (grabbed, frame) = capture.read()
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dims = autocrop(grayframe)
            if count == 0:
                smallest_dims = dims
            else:
                smallest_dims = [max(dims[0], smallest_dims[0]), max(dims[1], smallest_dims[1]), max(dims[2], smallest_dims[2]), max(dims[3], smallest_dims[3])]

        capture = cv2.VideoCapture("real_data/" + str(TRIAL) + "_vac_data.mp4")
        while True:
            (grabbed, frame) = capture.read()
            if not grabbed:
                break

            # frame = frame[smallest_dims[0]:smallest_dims[1], smallest_dims[2]:smallest_dims[3]]
            frame = cv2.resize(frame, (600, 600))
            # frame[:, :, 1] = 0
            # frame[:, :, 2] = 0

            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Credit: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
            # alpha = 5 # Contrast control (1.0-3.0)
            # beta = -50 # Brightness control (0-100)
            # alpha = 4 # Contrast control (1.0-3.0)
            # beta = -50 # Brightness control (0-100)
            # contrastframe = cv2.convertScaleAbs(grayframe, alpha=alpha, beta=beta)
            contrastframe = grayframe

            if TEST:
                cropped = contrastframe 
            else:
                cropped = contrastframe[TOP["1"]:BOT["1"], TOP["0"]:BOT["0"]] 

            cv2.imshow('Video', cropped) 
            trial.append(cv2.mean(cropped)[0])

            keyVal = cv2.waitKey(1) & 0xFF

        capture.release()
        cv2.destroyAllWindows()

        # double check 
        x = [x for x in range(len(trial))]
        y = trial 
        x_array = np.array(x)
        print(x, y)
        plot = sns.lineplot(x=x, y=y)
        plt.show(plot)
        dataset.append(trial)
    except KeyboardInterrupt:
        if TEST or len(sys.argv) > 1:
            pass
        else:
            break

np.save("dataset.pickle", dataset)
