import numpy as np
from scipy.misc import imsave
import os
import cPickle as pickle
import cv
import cv2

import matplotlib.pyplot as plt
plt.ion()

dataset_path = '/drogba/Datasets/ToyotaCar/'

file_lists = {'incidents': os.path.join(dataset_path, 'incidents.lst'),
        'near_incidents': os.path.join(dataset_path, 'nearincidents.lst')}

offsets = [40, 70, 100]

for key in file_lists:
    boundaries = pickle.load(open("boundary_{}.pkl".format(key)))

    contents = open(file_lists[key],'r').read().splitlines()
    
    if key == "incidents":
        directory = "Incidents"
    elif key == "near_incidents":
        directory = "Near Incidents"

    for video_name in contents:
        boundary = boundaries[video_name].tolist()
    
        cap = cv2.VideoCapture(os.path.join(dataset_path, directory, video_name))

        frames = []

        while cap.grab():
            frames.append(cap.retrieve()[1])

        boundary.append(len(frames)-1)

        print("{}: {}".format(key, video_name))

        for b in boundary:
            for offset in offsets:
                image_file = os.path.join("toyota_frames", 
                        key, "{}".format(offset), video_name[:-4] + ".jpg")
                imsave(image_file, frames[b-offset+1])

