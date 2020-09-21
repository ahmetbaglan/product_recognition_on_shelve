import pickle
import numpy as np

class Predictor_corr:
    def __init__(self, trainingName ):
        correlationOutDict = pickle.load( open( trainingName, "rb" ))
        self.centers = correlationOutDict['centers']
        self.numOfWords = correlationOutDict['numOfWords']
        self.cat = correlationOutDict['cat']
        self.maxVectorNumberFromImage = correlationOutDict['maxVecFromImage']


    def predict(self,i):
        distance = 0
        nowClass = 0
        for c in range(len(self.cat)):
            nowDistance = np.linalg.norm(np.dot(self.centers[self.cat[c]],np.array(i)))
            if nowDistance>distance:
                distance = nowDistance
                nowClass = c

        return nowClass