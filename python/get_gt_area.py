import numpy as np
import os
from scipy.io import loadmat
from scipy.misc import imread, imsave
import skimage.morphology as morphology
import matplotlib.pyplot as plt
plt.ion()

# top left: 0
# top right: 1
# bottom left: 2
# bottom right: 3
def is_inside(corner_locs, pt):
    tl, tr, bl, br = corner_locs

    y, x = pt
    
    #quick check
    if y <= min(tl[0], tr[0]):
        return False
    if x <= min(tl[1], bl[1]):
        return False
    if y >= max(bl[0], br[0]):
        return False
    if x >= max(tr[1], br[1]):
        return False
        
    #line checks
    #top
    diff = tr - tl
    norm = np.array([diff[1], -diff[0]])

    intercept = np.dot(tr, norm)
        
    if np.dot(pt, norm) <= intercept:
        return False
    
    #bottom
    diff = br - bl
    norm = np.array([diff[1], -diff[0]])
    intercept = np.dot(br, norm)
    
    if np.dot(pt, norm) >= intercept:
        return False
        
    #left
    diff = bl - tl
    norm = np.array([-diff[1], diff[0]])
    intercept = np.dot(tl, norm)    
    
    if np.dot(pt, norm) <= intercept:
        return False
        
    #right
    diff = tr - br
    norm = np.array([diff[1], -diff[0]])
    intercept = np.dot(tr, norm)
    
    if np.dot(pt, norm) >= intercept:
        return  False
        
    return True
    
    #np.dot(pt, norm)

annotation_dirs = [
    'C:/Users/Mert/SkyDrive/toyota_frames/incidents/labeled',
    'C:/Users/Mert/SkyDrive/toyota_frames/near_incidents/labeled']

output_dir = 'gt_area'
    
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
        
        corner_pix = np.logical_and(np.logical_and(
            im[:,:,0] < tolerance, 
            im[:,:,1] > 255 - tolerance), 
            im[:,:,2] < tolerance)
        
        region_labels = morphology.label(morphology.binary_dilation(corner_pix, np.ones((3,3))), neighbors=8)
    
        corner_locations = np.zeros((4, 2)).astype('float')
        for ci in range(1,5):
            locs = np.array(np.nonzero(region_labels==ci))
            corner_locations[ci-1, :] = locs.mean(axis=1)
    
        test = np.zeros((240, 352)).astype('bool')
            
        for i in range(240):
            for j in range(352):
                test[i,j] = is_inside(corner_locations, np.array([i, j]))
        
        imsave(output_file, test)
     
    ## Debug     
    #    plt.figure()
    #    plt.imshow(im)
    #    
    #    plt.figure()
    #    plt.imshow(region_labels)
    #    
    #    plt.figure()
    #    plt.imshow(test)
    #    
    #    break
    #break