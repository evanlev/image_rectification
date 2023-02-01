#!/anaconda/bin/python
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
from math import pi
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import numpy as np
import cv2

# ----- supporting functions for image rectification ---------
# INPUTS:
#   H      = perspective transformation (3x3 matrix)
#   limits = image boundaries
# OUTPUTS:
#   Htr,Hbr,Hbl = homogeneous 3-vectors = H * (image corners)
def getHCorners(H, limits):
    Ny = float(limits[0])
    Nx = float(limits[1])
    # Apply H to corners of the image to determine bounds
    Htr  = np.dot(H, np.array([0.0, Ny, 1.0]).flatten()) # Top left maps to here
    Hbr  = np.dot(H, np.array([Nx,  Ny, 1.0]).flatten()) # Bottom right maps to here
    Hbl  = np.dot(H, np.array([Nx, 0.0, 1.0]).flatten()) # Bottom left maps to here
    Hcor = [Htr,Hbr,Hbl]
    
    # Check if corners in the transformed image map to infinity finite
    finite = True 
    for y in Hcor:
        if y[2] == 0:
            finite = False

    return Hcor, finite

# INPUTS:
#   H      = perspective transformation (3x3 matrix)
#   limits = image boundaries
# OUTPUTS:
#   HS * H, where HS is an (isotropic) scaling to keep an image of shape 
#   limits contained within limits when HS*H is applied
def scaleHToImage(H, limits, anisotropic = False): # TODO: test anisotropic
    assert len(limits) >= 2 # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # If corners in the transformed image are not finite, don't do scaling
    if not finite:
        print("Skipping scaling due to point mapped to infinity")
        return H;
        
    # Maximum coordinate that any corner maps to
    k = [max([Hcor[j][i] / Hcor[j][2] for j in range(len(Hcor))])/float(limits[1-i]) for i in range(2)];

    # Scale
    if anisotropic:
        # print(f"Scaling by (%f,%f)\n" % (k[0], k[1])))
        HS = np.array([[1./k[0],0.0,0.0],[0.0,1./k[1],0.0],[0.0,0.0,1.0]])
    else:
        k = max(k)
        # print(("Scaling by %f\n" % k))
        HS = np.array([[1.0/k,0.0,0.0],[0.0,1.0/k,0.0],[0.0,0.0,1.0]])

    return np.dot(HS, H)

# INPUTS:
#   H      = projective transformation
#   line   = line in the source image (domain of H)
# OUTPUTS:
#   HR * H = New H where HR is a rotation chosen to make line map to 
#            either vertical or horizontal, chosen 
def rotateHToLine(H, line):
    assert len(line) == 3
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Compute transformed line = H^-T * l
    lineTr = np.dot(np.linalg.inv(H).T, line)

    # Rotate so that this line is horizonal in the image 
    r1 = np.array([lineTr[1], -lineTr[0]]) # First row of R is perpendicular to linesTr[0]
    # print("-----------", r1.flatten())
    r1 = r1 / np.linalg.norm(r1.flatten())
    theta = np.arctan2(-r1[1] , r1[0])
    if abs(theta) < pi/4:
        R = np.array([[r1[0],  r1[1]], [-r1[1], r1[0]]])
    else:
        R = np.identity(2)
        #R = np.array([[r1[1], -r1[0]], [ r1[0], r1[1]]])
    theta = np.arctan2(R[1,0], R[1,1])
    # print(("Rotating by %.1f degrees" % (theta*180/pi)))
    HR = np.identity(3)
    HR[0:2,0:2] = R

    return np.dot(HR,H)

# INPUTS:
#   H      = projective transformation (3x3)
#   limits = image size
# OUTPUTS:
#   HT*H   = new projective transformation such that HT is a translation and 
#            HT*H x > 0 for all x > 0
def translateHToPosQuadrant(H, limits):
    assert len(limits) >= 2 # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # Check if corners map to infinity, if so skip translation
    if not finite:
        print("Corners map to infinity, skipping translation")
        return H

    # Min coordinates of H * image corners
    minc = [min([Hcor[j][i]/Hcor[j][2] for j in range(len(Hcor))]) for i in range(2)]

    # Choose translation
    HT = np.identity(3)
    HT[0,2] = -minc[0]
    HT[1,2] = -minc[1]

    return np.dot(HT, H)


def translateAndScaleHToImage(H, limits, anisotropic = False):
    H = translateHToPosQuadrant(H, limits)
    H = scaleHToImage(H, limits, anisotropic)
    return H


# Get two mouse clicks from the user with ginput to select points
# return coordinates of clicked points (x1[i],y1[i]) and line through the
# pair of points, a non-normalized (homogeneous) 3-vector
def getLine():
    # get mouse clicks
    pts = []
    while len(pts) == 0: # FIXME
        pts = ginput(n=2)
    pts_h = [[x[0],x[1],1] for x in pts]
    line = np.cross(pts_h[0], pts_h[1]) # line is [p0 p1 1] x [q0 q1 1]
    # return points that were clicked on for plotting
    x1=[x[0] for x in pts] # map applies the function passed as 
    y1=[x[1] for x in pts] # first parameter to each element of pts
    return x1,y1,line

# INPUTS:
#   A = matrix
# OUTPUTS:
#   N = basis for the null-space
#
# Example: nullspace(np.array([[1,0,0],[0,1,0]])) returns [0,0,1]
def nullspace(A, eps=1e-15):
    u, s, vh = sp.linalg.svd(A,full_matrices=1,compute_uv=1)
    # Pad so that we get the nullspace of a wide matrix. 
    N = A.shape[1]
    K = s.shape[0]
    if K < N:
        s[K+1:N] = 0
        s2 = np.zeros((N))
        s2[0:K] = s
        s = s2
    null_mask = (s <= eps)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)

# return smallest singular vector of A (or the nullspace if A is 2x3)
def smallestSingularVector(A):
    if A.shape[0] == 2 and A.shape[1] == 3:
        return nullspace(A)
    elif(A.shape[0] > 2):
        u,s,vh = sp.linalg.svd(A,full_matrices=1,compute_uv=1)
        vN = vh[vh.shape[0]-1,:]
        vN = vN.conj().T
        return vN
    else:
        raise Exception("bad shape of A: %d %d" % (A.shape[0], A.shape[1]))


# INPUTS:
#   im = image (2D or 3D array for 1 or 3 color channels)
#   H  = transformation
# OUTPUTS:
#   im2 = H applied to im
def myApplyH(im, H):
    return cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))

# INPUTS: 
#   im = image
# OUTPUTS:
#   image with outer regions cropped
def cropOuterRegion(im):
    # Find upper limit on image
    bottom = im.shape[0]
    top  = 0
    right = im.shape[1]
    left  = 0
    while left+1 < im.shape[1] and max(im[:,left+1].flatten()) == 0:
        left = left + 1
    while right >= 1 and max(im[:,right-1].flatten()) == 0:
        right = right - 1
    while bottom >= 1 and max(im[bottom-1,:].flatten()) == 0:
        bottom = bottom - 1
    while top+1 < im.shape[0] and max(im[top+1,:].flatten()) == 0:
        top = top + 1
    if im.ndim == 3:
        imc = im[top:bottom,left:right,:]
    elif im.ndim == 2:
        imc = im[top:bottom,left:right]
    else:
        raise Exception("TODO: reshape")
    return imc

# INPUT:
#   size = image size
#   l    = line (homogeneous 3-vector)
# OUTPUTS:
#   xx,yy = plot(xx,yy) will plot the line cropped within the image region
def getPlotBoundsLine(size, l):
    l = l.flatten()
    L = 0
    R = 1
    T = 2
    B = 3
    Nx = size[1]
    Ny = size[0]
    # lines intersecting image edges
    lbd = [[] for x in range(4)]
    lbd[L] = np.array([1.0, 0.0, 0.0])
    lbd[R] = np.array([1.0, 0.0, -float(Nx)])
    lbd[T] = np.array([0.0, 1.0, 0.0])
    lbd[B] = np.array([0.0, 1.0, -float(Ny)])
    I = [np.cross(l, l2) for l2 in lbd]

    # return T/F if intersection point I is in the bounds of the image
    Ied = [] # List of (x,y) where (x,y) is an intersection of the line with the boundary
    for i in [L, R]:
        if I[i][2] != 0:
            In1 = I[i][1] / I[i][2]
            if In1 > 0 and In1 < Ny:
                Ied.append(I[i][0:2]/I[i][2])

    for i in [T, B]:
        if I[i][2] != 0:
            In0 = I[i][0] / I[i][2]
            if In0 > 0 and In0 < Nx:
                Ied.append(I[i][0:2]/I[i][2])

    assert(len(Ied) == 2 or len(Ied) == 0)
    xx = [Ied[x][0] for x in range(0,len(Ied))]
    yy = [Ied[x][1] for x in range(0,len(Ied))]

    return xx,yy

# INPUTS:
#   im               = image
#   nLinePairs       = number of line pairs to use
#   doRotationAfterH = (T/F) do rotation after removing 
#                      projective distortion 
#   doScalingAfterH  = (T/F) scale output to image bounds 
# OUTPUTS:
#   imRect  = image with projective distortion removed
def rectifyAffineF(im, nLinePairs, doRotationAfterH = True, doTranslationAfterH = True, doScalingAfterH = True):
    # --------- Supporting functions -----------
    # Plot image, lines, vanishing points, vanishing line
    def replotAffine(im,limits,lines=[[],[]],x=[[],[]],y=[[],[]],vPts=[]):
        # -- Settings for this function ---
        plot_lines  = True
        plot_vpts   = True
        plot_points = True
        plot_vline  = True
        # ---------------------------------
        if len(x) != len(y):
            raise Exception("len(x): %d, len(y): %d!" % (len(x), len(y)))
        if len(lines) != len(x):
            raise Exception("len(x): %d, len(lines): %d!" % (len(x), len(lines)))
        if len(vPts) != min(len(x[0]), len(x[1])):
            raise Exception("len(x[0]): %d, len(x[1]): %d, len(vpts): %d!" % (len(x[0]), len(x[1]), len(vPts)))

        plt.close() # ginput does not allow new points to be plotted
        imshow(im,cmap='gray')
        axis('image')
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Determine how many lines to plot in red, leaving the last in green if the second needs to be picked
        nl1 = len(y[0])
        nl2 = len(y[1])
        if nl1 == nl2:
            nred = nl1
        else:
            nred = nl1 - 1
        # Plot extension of user-selected lines (dashed)
        if plot_lines:
            for k in range(nred):
                xx,yy = getPlotBoundsLine(limits, lines[0][k])
                plot(xx,yy,'r--')
            if nl1 - nred > 0:
                xx,yy = getPlotBoundsLine(limits, lines[0][nl1-1])
                plot(xx,yy,'g--')
            for l in lines[1]:
                xx,yy = getPlotBoundsLine(limits, l)
                plot(xx,yy,'b--')

        # Plot user-selected line segments (solid)
        if plot_points:
            # Plot lines: direction 1, all red but the last one green
            for k in range(0,nred):
                plot(x[0][k],y[0][k],'r-')
            if nl1 - nred > 0:
                plot(x[0][nl1-1],y[0][nl1-1],'g-')
            # Plot lines: direction 2
            for k in range(0,len(y[1])):
                plot(x[1][k],y[1][k],'b-')

        # Compute normalized vanishing points for plotting
        vPts_n = [[0,0] for x in vPts]
        vPtInImage = [True for x in vPts]
        for i in range(len(vPts)):
            if vPts[i][2] == 0:
                vPtInImage[i] = False
            else:
                vPts_n[i][0] = vPts[i][0] / vPts[i][2]
                vPts_n[i][1] = vPts[i][1] / vPts[i][2]
                vPtInImage[i] = vPts_n[i][0] < limits[0] and vPts_n[i][0] > 0 and vPts_n[i][1] < limits[1] and vPts_n[i][1] > 0

        # Plot vanishing points 
        if plot_vpts:
            for i in range(len(vPts_n)):
                if vPtInImage[i]:
                    plot(vPts_n[i][0], vPts_n[i][1], 'yo')
        # Plot vanishing line
        if plot_vline:
            if len(vPts) == 2 and vPtInImage[0] and vPtInImage[1]:
                vLine = np.cross(vPts[0], vPts[1])
                xx,yy = getPlotBoundsLine(limits, vLine)
                plot(xx,yy,'y-')
                #plot([vPts_n[0][0],vPts_n[1][0]], [vPts_n[0][1],vPts_n[1][1]], 'y-')

        # Limit axes to the image
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])


    # -------------- START -----------------

    # Create figure
    fig2 = plt.figure()
    replotAffine(im,im.shape)

    # Lines
    lines = [[],[]]
    x = [[],[]]
    y = [[],[]]
    vPts = []
        
    # Get line pairs interactively
    for i in range(0,2*nLinePairs):
        ii = i % 2
        if ii == 1:
            plt.suptitle('Click two points intersecting a line parallel to the green line')
        else:
            if i == 0:
                plt.suptitle('Click two points intersecting the first of two parallel lines')
            else:
                plt.suptitle('Click two points intersecting the first of two parallel lines not parallel to the first set')
        x1,y1,line = getLine()
        x[ii].append(x1)
        y[ii].append(y1)
        lines[ii].append(line)
        if ii == 1:
            nlp = len(lines[0])
            vPt = np.cross(lines[0][nlp-1], lines[1][nlp-1])
            if vPt[2] != 0.:
                vPt[0] = vPt[0] / vPt[2]
                vPt[1] = vPt[1] / vPt[2]
                vPt[2] = vPt[2] / vPt[2]
            vPts.append(vPt)
        # re-plot figure
        replotAffine(im,im.shape,lines,x,y,vPts)

    print("Vanishing points:")
    print((vPts[0]))
    print((vPts[1]))
    vLine = np.cross(vPts[0], vPts[1])
    print("Vanishing line:")
    print(vLine)
    H = np.identity(3)
    H[2,0] = vLine[0] / vLine[2]
    H[2,1] = vLine[1] / vLine[2]

    print("H:")
    print(H)

    # Rotate after doing H 
    if doRotationAfterH:
        H = rotateHToLine(H, lines[0][0])

    # Translate to keep Hx > 0
    if doTranslationAfterH:
        H = translateHToPosQuadrant(H, im.shape)

    # Scale to keep the output contained just within the image matrix
    if doScalingAfterH:
        H = scaleHToImage(H, im.shape, False)

    # Apply H to do affine rectification
    imRect = myApplyH(im, H)

    plt.close(fig2)
    return imRect, H

# INPUTS:
#   im         = image with projective distortion removed
#   nLinePairs = number of line pairs to use
#   doRotationAfterH = (T/F) choose a rotation for the output
#   doScalingAfterH  = (T/F) scale output to image bounds 
# OUTPUTS:
#   imRect  = image with affine distortion removed
def rectifyMetricF(imA, nLinePairs, doRotationAfterH = True, doTranslationAfterH = True, doScalingAfterH = True):
    # --------- Supporting functions -----------
    # Plot image and lines
    def replotMetric(imA,limits,lines=[[],[]],x=[[],[]],y=[[],[]]):
        plt.close() # ginput does not allow new points to be plotted
        imshow(imA,cmap='gray')
        axis('image')

        # Plot settings 
        plot_lines = False
        plot_points = True
        
        # Determine how many lines to plot in red, leaving the last in green if the second needs to be picked
        nl1 = len(y[0])
        nl2 = len(y[1])
        if nl1 == nl2:
            nred = nl1
        else:
            nred = nl1 - 1
        if plot_lines:
            for k in range(nred):
                xx,yy = getPlotBoundsLine(limits, lines[0][k])
                plot(xx,yy,'r--')
            if nl1 - nred > 0:
                xx,yy = getPlotBoundsLine(limits, lines[0][nl-1])
                plot(xx,yy,'g--')
            for l in lines[1]:
                xx,yy = getPlotBoundsLine(limits, l)
                plot(xx,yy,'b--')
        if plot_points:
            # Plot lines: direction 1, all red but the last one green
            for k in range(0,nred):
                plot(x[0][k],y[0][k],'r-')
            if nl1 - nred > 0:
                plot(x[0][nl1-1],y[0][nl1-1],'g-')
            # Plot lines: direction 2
            for k in range(0,len(y[1])):
                plot(x[1][k],y[1][k],'b-')
        axis('off')
        axis('image')
        draw()


    # m = list of lines
    # l = list of lines orthogonal to lines in m
    #
    # Return S = AA', where A is the affine part of the transformation
    def linesToS(l, m, scale_opt=0):
        if len(l) != len(m):
            raise Exception("lenl: %d, len(m): %d", len(l), len(m))
        nlines = len(l)

        if scale_opt == 1:
            # Solve for S = [s11 s21  s21 s22], then scale by 1/max(S)
            A = np.zeros((len(l), 3))
            for r in range(0,len(l)):
                A[r,0] = l[r][0]*m[r][0]
                A[r,1] = l[r][0]*m[r][1] + l[r][1]*m[r][0]
                A[r,2] = l[r][1]*m[r][1]

            s = smallestSingularVector(A) 
            S = np.array([[s[0], s[1]], [s[1], s[2]]]).reshape((2,2))
            S = S / max(s[0],s[2])
        else:
            # Solve for S = [s11 s21  s21 1]
            RHS = np.zeros((nlines,1))
            A = np.zeros((nlines, 2))
            for k in range(0,nlines):
                RHS[k] = -l[k][1]*m[k][1]
            for k in range(0,nlines):
                A[k,0] = l[k][0]*m[k][0]
                A[k,1] = l[k][0]*m[k][1] + l[k][1]*m[k][0]
            s = np.linalg.lstsq(A,RHS)[0]
            S = np.array([[s[0][0], s[1][0]],[s[1][0],1]])
        return S

    # -------------- START -----------------
    
    # Create figure
    fig1 = plt.figure()
    spd = False
    while not spd:
        replotMetric(imA,imA.shape)
        
        # Lines
        lines = [[],[]] # lines[i][j][k] contains the line for the i'th direction (0,1), j'th selected line, k'th coordinate (0,1,2)
        x = [[],[]]     # x[i][j][k] contains x coordinate for the i'th direction (0,1), j'th selected line, k'th (0,1) selected point
        y = [[],[]]     # y[i][j][k] contains y coordinate for the i'th direction (0,1), j'th selected line, k'th (0,1) selected point

        # Get line pairs interactively
        for i in range(0,2*nLinePairs):
            ii = i % 2
            if ii == 1:
                plt.suptitle('Click two points intersecting a line perpendicular to the green line')
            else:
                if i == 0:
                    plt.suptitle('Click two points intersecting the first of two perpendicular lines')
                else:
                    plt.suptitle('Click two points intersecting the first of two perpendicular lines not parallel to any in the first set')
            x1,y1,line = getLine()
            x[ii].append(x1)
            y[ii].append(y1)
            lines[ii].append(line)
            # re-plot figure
            replotMetric(imA,imA.shape,lines,x,y)

        # Form S = KK'
        S = linesToS(lines[0], lines[1])

        # lprime = H^-T l
        u,s,vh = sp.linalg.svd(S,full_matrices=1,compute_uv=1)
            
        try: # FIXME
            A = np.linalg.cholesky(S)
            spd = True;
        except np.linalg.linalg.LinAlgError:
            w = np.linalg.eig(S)
            print('S was not SPD, try again...')
            print("S = ")
            print(S)
            print("eigenvalues: ")
            print(w)
            continue;

    # form H from A
    H = np.zeros((3,3))
    H[0:2,0:2] = A
    H[2,2] = 1
    print("H:")
    print(H)
    Hinv = np.linalg.inv(H)
    if False: # not the best way to set the scaling
        Hinv = Hinv / max(Hinv[0:2,0:2].flatten())
        print("Hinv:")
        print(Hinv)
        Hinv[2,2] = 1
        print("-> Hinv:")
        print(Hinv)

    # Rotate to make linesTr0 parallel to either [0 1 0] or [1 0 0]  
    # i.e.  (HR^-T * linesTr0) = [0 C D]^T, any D
    # Note HR^-T = HR
    # r1 * linesTr0 = 0
    if doRotationAfterH:
        Hinv = rotateHToLine(Hinv, lines[0][0])
     
    # Translate to keep Hx > 0
    if doTranslationAfterH:
        Hinv = translateHToPosQuadrant(Hinv, imA.shape)

    # Scale to keep the output contained just within the image matrix
    if doScalingAfterH:
        Hinv = scaleHToImage(Hinv, imA.shape, False)
  
    # Do rectification
    imRect = myApplyH(imA, Hinv)

    # Clean up
    plt.close(fig1)

    # Return rectified image
    return imRect, Hinv

