import os
import numpy as np
import cv2
import cPickle as pickle

import matplotlib.pyplot as plt
plt.ion()

from scipy.misc import imsave

dataset_path = '/drogba/Datasets/ToyotaCar/'

file_lists = {'incidents': os.path.join(dataset_path, 'incidents.lst'),
        'near_incidents': os.path.join(dataset_path, 'nearincidents.lst')}

offset = 70

out_dir = 'frg/'

for key in file_lists:
    boundaries = pickle.load(open("boundary_{}.pkl".format(key)))

    contents = open(file_lists[key],'r').read().splitlines()
    
    if key == "incidents":
        directory = "Incidents"
    elif key == "near_incidents":
        directory = "Near Incidents"

    for video_name in contents:
        print("{}: {}".format(directory, video_name))
        
        fg_file1 = os.path.join(out_dir, "{}.png".format(video_name[:-4]))

        if os.path.exists(fg_file1):
            continue

        boundary = boundaries[video_name].tolist()
    
        cap = cv2.VideoCapture(os.path.join(dataset_path, directory, video_name))

        bgs1 = cv2.BackgroundSubtractorMOG(10, 1, 0.9)

        foreground1 = []

        fc = 0

        while cap.grab():
            fr = cap.retrieve()[1]

            if len(boundary) == 0 or fc > boundary[-1]:
                foreground1.append(bgs1.apply(fr))

            fc += 1

        imsave(fg_file1, foreground1[len(foreground1) - offset])
