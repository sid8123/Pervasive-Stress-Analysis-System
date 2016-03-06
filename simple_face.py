import cv2
import sys
import numpy

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = cv2.imread(imagePath)
#image *= 1./255;
#luv=cv2.cvtColor(image,CV_RGB2LUV)
#Convert from BR to Gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
cv2.imshow("Faces found",image)
cv2.imshow("Faces1",region)
cv2.waitKey(0)
