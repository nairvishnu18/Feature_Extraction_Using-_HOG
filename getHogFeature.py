import cv2

class HOG:
    def __init__(self,bins):
        self.bins = bins
    def getHog(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image],[0,1,2],None,[self.bins,self.bins,self.bins],[0,256,0,256,0,256])
        cv2.normalize(hist,hist)
        return hist.flatten()


