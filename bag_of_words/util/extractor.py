import cv2
import numpy as np
from sklearn.cluster import KMeans

from bagOfWord.util.imageRetriever import *
from bagOfWord.util.util import *

class featureExtractor:
    def __init__(self, cat,maxVectorNumberFromImage,numOfWords,isTrain = True, kmeans = None):
        if isTrain:
            A = 'train'
        else:
            A = 'test'
        self.cat = cat
        self.maxVectorNumberFromImage = maxVectorNumberFromImage
        self.numOfWords = numOfWords
        self.nowR = imageRetriever(cat)
        self.imagesAdresses = self.nowR.getDictOfDir(A)
        self.sift = cv2.SIFT()
        self.kmeans = kmeans
        self.classCenterFeatures = None


    def getFeatures(self):
        allVectors = {}
        for i in self.cat:
            allVectors[i] = [[] for x in xrange(len(self.imagesAdresses[i]))]

        for k in self.imagesAdresses:
            imNo = 0;
            for j in self.imagesAdresses[k]:
                try:
                    nowImage = cv2.imread(j)
                    if(type(nowImage) != np.ndarray):
                        raise ValueError('The image could not be read')
                except:
                    continue

                nowImageGray = cv2.cvtColor(nowImage, cv2.COLOR_BGR2GRAY)
                kp = self.sift.detect(nowImageGray, None)
                siftOut = self.sift.compute(nowImageGray, kp)
                imageVectors = siftOut[1]
                till = min(self.maxVectorNumberFromImage, len(imageVectors))
                # np.random.shuffle(imageVectors)
                my_sift_vectors = imageVectors[0:till]
                allVectors[k][imNo] = my_sift_vectors
                imNo+=1


        if self.kmeans == None:

            cumVectors = []

            for i in allVectors:
                for j in allVectors[i]:
                    cumVectors.extend(j)
            self.kmeans = KMeans(n_clusters=self.numOfWords, random_state=0).fit(cumVectors)

        featuresDict = {}

        for i in self.cat:
            featuresDict[i] = []

        for i in allVectors:
            for imgVe in allVectors[i]:
                a = self.kmeans.predict(imgVe)
                b = getHistogram(a,self.numOfWords)
                featuresDict[i].append(b)

        self.classCenterFeatures = {}
        for i in featuresDict:
            c = np.zeros(self.numOfWords)
            for img in featuresDict[i]:
                c += np.array(img)
            self.classCenterFeatures[i] = c / len(featuresDict[i])



        return featuresDict

    def getKmeans(self):
        return self.kmeans

    def getClassCenterFeatures(self):
        return self.classCenterFeatures

    def sumlist(self,a,b):
        a = np.array(a)
        b = np.array(b)
        return a+b

if __name__ == '__main__':
    cat = ['bagdat1']
    maxVectorNumberFromImage= 100000
    numOfWords= 20
    e = featureExtractor(cat,maxVectorNumberFromImage,numOfWords)
    print e.getFeatures()

