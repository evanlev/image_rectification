#!/anaconda/bin/python
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from utils import *

# ---------- Settings ---------
nLinePairs = 2; # Select this may pairs of perpendicular lines
# -----------------------------
def usage(sname):
    print(sname + " <image>");

if len(sys.argv) != 2:
    usage(sys.argv[0]);
    exit(0);

# print(f"Two-stage (affine and metric) rectification using  {d} parallel line pairs {nLinePairs}");

# Read input files
imgPath = sys.argv[1];
fileparts = os.path.splitext(imgPath);
if fileparts[1] == '':
    fileparts = (fileparts[0], '.png');
imgPath = fileparts[0] + fileparts[1];
im = mplimg.imread(imgPath);

# Do rectification in two stages
imA, HA = rectifyAffineF(im,  nLinePairs)
plt.close('all')
imM, HM = rectifyMetricF(imA, nLinePairs)

# Translate and scale HM * HA to the image
H = translateAndScaleHToImage(np.dot(HM,HA), imA.shape)

# Apply translated and scaled version of HM * HA
imM = myApplyH(im, H) # Chain the two transforms
plt.close('all')

# Show result
imshow(np.concatenate((im,imA,imM),axis=1));
axis('off')
plt.suptitle('Original (left), Affine rectified (middle), Metric rectification (right)')
margin = 0.05;
plt.subplots_adjust(left=margin, right=1.0-margin, top=1.0-margin, bottom=margin)
show()

# Save output
imgPathRect = fileparts[0] + "_rect" + fileparts[1];
# print(f"Saving output to {s}... {imgPathRect}");
mplimg.imsave(imgPathRect,cropOuterRegion(imM));






