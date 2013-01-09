import numpy as np
import os
from scipy.io import loadmat

from scipy.misc import imread, imsave

import matplotlib.pyplot as plt
#plt.ion()

area_files = os.listdir('gt_area')

gt_areas = []
gts = []
rpca_results = []
gmm_results = []

for f in area_files:
    if f[-3:] != 'png':
        continue
    gt_areas.append(imread(os.path.join('gt_area', f)) > 0)
    gts.append(imread(os.path.join('gt_labels', f)) > 0)

    base_name = f[:-4]

    #load the result
    path1 = os.path.join(
           'RPCA_ICIP_FRAMES', 'Incidents', base_name, 'confid_maps.mat')

    path2 = os.path.join(
           'RPCA_ICIP_FRAMES', 'Near Incidents', base_name, 'confid_maps.mat')

    if os.path.exists(path1):
        rpca_conf = loadmat(path1)
    else:
        rpca_conf = loadmat(path2)

    rpca_results.append(np.abs(rpca_conf['confid_w']).mean(axis=2))

    gmm_path = os.path.join(
            'frg', '{}.png'.format(base_name))

    gmm_results.append(imread(gmm_path) > 0)

    #side by side
    gmm_tp = np.logical_and(gt_areas[-1], np.logical_and(gmm_results[-1], gts[-1]))
    gmm_fp = np.logical_and(gt_areas[-1], np.logical_and(gmm_results[-1], np.invert(gts[-1])))

    true_gt = np.logical_and(gt_areas[-1], gts[-1])

    gmm_res_im = np.zeros((gts[-1].shape[0], gts[-1].shape[1], 3)).astype('float')
    gmm_res_im[:,:,0] = gmm_tp.astype('float')
    gmm_res_im[:,:,1] = true_gt.astype('float')
    gmm_res_im[:,:,2] = gmm_fp.astype('float')

    rpca_tp = np.logical_and(gt_areas[-1], np.logical_and(rpca_results[-1] > 0.5, gts[-1]))
    rpca_fp = np.logical_and(gt_areas[-1], np.logical_and(rpca_results[-1] > 0.5, np.invert(gts[-1])))

    rpca_res_im = np.zeros((gts[-1].shape[0], gts[-1].shape[1], 3)).astype('float')
    rpca_res_im[:,:,0] = rpca_tp.astype('float')
    rpca_res_im[:,:,1] = true_gt.astype('float')
    rpca_res_im[:,:,2] = rpca_fp.astype('float')
   

    out_im = np.hstack((gmm_res_im, rpca_res_im))

    imsave(os.path.join('comparison', base_name + '.png'), out_im)

gt_areas = np.vstack(gt_areas)
gts = np.vstack(gts)
rpca_results = np.vstack(rpca_results)
gmm_results = np.vstack(gmm_results)

interval = np.linspace(rpca_results.min(), rpca_results.max(), num=100)

tp = []
fp = []

pos = np.logical_and(gt_areas, gts)
neg = np.logical_and(gt_areas, np.invert(gts))

n_pos = float(pos.sum())
n_neg = float(neg.sum())

gmm_tp = np.logical_and(gmm_results, pos).sum() / n_pos
gmm_fp = np.logical_and(gmm_results, neg).sum() / n_neg

plt.plot(gmm_fp, gmm_tp, 'ro', label='GMM Result')

for i in interval[:-1]:
    thresh = rpca_results > i
    true_pos = np.logical_and(thresh, pos).sum()
    false_pos = np.logical_and(thresh, neg).sum()

    tp.append(float(true_pos) / n_pos)
    fp.append(float(false_pos) / n_neg)

plt.plot(np.array(fp), np.array(tp), label='RPCA Result')

plt.legend()
plt.xlim([0, 0.2])
plt.xlabel('False Positives')
plt.ylabel('True Positives')

fig = plt.gcf()
fig.savefig('toyota_roc.pdf')
