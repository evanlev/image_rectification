# image_rectification
Interactive affine and metric image rectification in Python. Click coplanar parallel and perpendicular lines on any image to rectify it, displaying and saving the output. Three scripts are provided for this.

![Screenshot](image_rectification.png)

## rectifyAffine.py
Select two parallel lines in an image and do affine rectification via the vanishing line following Fig. 2.13 in Hartley and Zisserman. Images are rotated to keep the first selected line horizontal and translated and scaled to keep the warped image contained just within the image matrix.

## rectifyMetric.py
Select two perpendicular lines in an image and do affine rectification via the vanishing line following example 2.26 in Hartley and Zisserman. Images are translated and scaled to keep the warped image contained just within the image matrix.

## rectify.py
Do affine followed by metric rectification. Images are translated and scaled once after both stages to keep the image contained just within the image matrix.

## Requirements
OpenCV Python, matplotlib, numpy, scipy

### References
Hartley, Richard, and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.
