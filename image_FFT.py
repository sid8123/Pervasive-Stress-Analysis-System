import cv2
import sys
import numpy
from scipy import fftpack
import pyfits
import numpy as np
import pylab as py
import radialProfile
import matplotlib.pyplot as plt
# Data from command line
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = cv2.imread(imagePath)
#image *= 1./255;
#luv=cv2.cvtColor(image,CV_RGB2LUV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#
#cv2.cvtColor(image,image,cv2.COLOR_BGR2GRAY);

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    region = image[y:y+h,x:x+w]
    # skin detection
    face_ycbcr = cv2.cvtColor(region, cv2.COLOR_BGR2YCR_CB)

    #mask limit 
    l_limit = numpy.array ([30,133,77])
    u_limit = numpy.array ([9999,173,127])

    mask = cv2.inRange (face_ycbcr, l_limit, u_limit)
    mask_inv = cv2.bitwise_not(mask)
    face_ycbcr = cv2.bitwise_and (face_ycbcr, face_ycbcr, mask=mask)
    skin = cv2.cvtColor (face_ycbcr, cv2.COLOR_YCR_CB2BGR)	
    skin_luv = cv2.cvtColor (skin, cv2.COLOR_BGR2LUV)

# Permutation of color space
face_luv = cv2.cvtColor(region, cv2.COLOR_BGR2LUV)

fh, fw, fd = face_luv.shape
face_u = numpy.zeros((fh, fw, fd), numpy.uint8)

_, u, _ = cv2.split(face_luv)
face_u[:,:,1] = u

# u* frame with skin detection
        
region = cv2.bitwise_and (skin_luv, face_u)     

F1 = fftpack.fft2(u)
 
# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fftpack.fftshift( F1 )
 
# Calculate a 2D power spectrum
psd2D = np.abs( F2 )**2
 
# Calculate the azimuthally averaged 1D power spectrum
psd1D = radialProfile.azimuthalAverage(psd2D)
 
# Plot
py.figure(1)
py.clf()
py.imshow( np.log10(u ), cmap=py.cm.Greys)
 
py.figure('2D Power Spectra of image')
py.clf()
py.imshow( np.log10( psd2D ))

py.figure('1D Power Spectra of image')

py.clf()
py.semilogy( psd1D )
py.xlabel('Frequency')
py.ylabel('Power spectrum')

py.show()   

cv2.waitKey(0)
