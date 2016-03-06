import cv2
import sys
import numpy
import numpy as np
# Get user supplied values
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

print "Found {0} faces".format(len(faces))

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
    #cv2.imshow("Faces found3",green)
    skin_luv = cv2.cvtColor (skin, cv2.COLOR_BGR2LUV)

face_luv = cv2.cvtColor(region, cv2.COLOR_BGR2LUV)

fh, fw, fd = face_luv.shape
face_u = numpy.zeros((fh, fw, fd), numpy.uint8)

_, u, _ = cv2.split(face_luv)
face_u[:,:,1] = u

        
# u* frame with skin detection
        
face = cv2.bitwise_and (skin_luv, face_u)

cv2.imshow('Skin Detection', skin)
cv2.imshow('Permutation of ColorSpace', face_luv)
cv2.imshow('Final Image', region)
cv2.imshow("Faces found",image)
cv2.waitKey(0)
