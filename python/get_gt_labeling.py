import numpy as np
import os
from scipy.io import loadmat
from scipy.misc import imread, imsave
import skimage.morphology as morphology
import matplotlib.pyplot as plt
plt.ion()

annotation_dirs = [
    'C:/Users/Mert/SkyDrive/toyota_frames/incidents/labeled',
    'C:/Users/Mert/SkyDrive/toyota_frames/near_incidents/labeled']

output_dir = 'gt_labels'
    
file_list = []

tolerance = 35

for d in annotation_dirs:
    annotated_files = os.listdir(d)

    for f in annotated_files:
        if f[-3:] != 'png':
            continue
        
        print("{}".format(f))
            
        file_name = os.path.join(d, f)

        output_file = os.path.join(output_dir, f)
        
        if os.path.exists(output_file):
            continue

        im = imread(file_name)
                
        gt_pix = np.logical_and(np.logical_and(
            im[:,:,0] < tolerance, 
            im[:,:,1] < tolerance), 
            im[:,:,2] > 255 - tolerance)

        gt_pix = morphology.closing(gt_pix, np.ones((3,3)))
        
        imsave(output_file, gt_pix)