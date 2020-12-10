#Dataset INRIA Dataset 
#Contain positive AND NEGATAIVE IMAGES
#POS  - ALL HUMANS
#NEG - OTHER THAN HUMANS IMAGES

from getHogFeature import HOG
from imutils import paths
import cv2
import argparse
import os
from sklearn.svm import LinearSVC



ap = argparse.ArgumentParser()
ap.add_argument("-t","--train", required="True", help="Path to training Images")
ap.add_argument("-e","--test",required="True",help="Path to the test images")
args = vars(ap.parse_args())

desc =HOG(9)
feature = []
labels = []

#Inside train - Human and Not Human Folders must be there
# Human Folder shouldl contain all images of humans
#Not Human should contain all other images except humans


for imagePath in paths.list_images(args["train"]):
    print(imagePath)
    image = cv2.imread(imagePath)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = desc.getHog(img)

    labels.append(imagePath.split(os.path.sep)[-2])
    feature.append(hist)

clf = LinearSVC(C=100.0 ,random_state=42)
clf.fit(feature,labels)


#Test should contain all the images to be tested
for imagePath in paths.list_images(args["test"]):
    image = cv2.imread(imagePath)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = desc.getHog(img)
    prediction = clf.predict(hist.reshape(1,-1))

    cv2.putText(image,prediction[0],(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
    cv2.imshow('HOG',image)
    cv2.waitKey(0)

