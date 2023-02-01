#!/anaconda/bin/python
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
from pylab import show, axis, imshow, draw
import os
import sys
from utils import *

# Apply a metric rectification to an image using pairs of perpendicular lines

# ---------- Settings ---------
nLinePairs = 2; # Select this may pairs of perpendicular lines
# -----------------------------

def usage(sname):
    print(sname + " <affine-rectified image> <# line pairs (>1)>");

if len(sys.argv) != 2 and len(sys.argv) != 3:
    usage(sys.argv[0]);
    exit(0);

if len(sys.argv) == 3:
    nLinePairs = int(sys.argv[2]);
    if nLinePairs < 2:
        usage(sys.argv[0]);

print(("Metric rectification using %d perpendicular line pairs" % nLinePairs));
    
# Read input files
imgPath = sys.argv[1];
fileparts = os.path.splitext(imgPath);
if fileparts[1] == '':
    fileparts = (fileparts[0], '.png');
imgPath = fileparts[0] + fileparts[1];
im = mplimg.imread(imgPath);

# Do metric rectification
imRect, HM = rectifyMetricF(im, nLinePairs);

# Show output
imshow(np.concatenate((im,imRect),axis=1),cmap='gray')
axis('image')
plt.suptitle('Original (left), Rectified (right)')
margin = 0.05;
plt.subplots_adjust(left=margin, right=1.0-margin, top=1.0-margin, bottom=margin)
show()
    
# Save output
imgPathRect = fileparts[0] + "_mrect" + fileparts[1];
print(("Saving output to %s..." % imgPathRect));
mplimg.imsave(imgPathRect,cropOuterRegion(imRect));


