from bagOfWord.predict.correlation_predict import *
from bagOfWord.util.extractor import *
import os
trainingName = './models/fall_correlation'

correlationOutDict = pickle.load( open( trainingName, "rb" ))
kmeans = correlationOutDict['kmeans']
centers = correlationOutDict['centers']
numOfWords = correlationOutDict['numOfWords']
cat = correlationOutDict['cat']
maxVectorNumberFromImage = correlationOutDict['maxVecFromImage']


testExtractor = featureExtractor(cat,maxVectorNumberFromImage,numOfWords,False, kmeans)
testFeatureDict = testExtractor.getFeatures()

p = Predictor_corr(trainingName)

classNo = 0.0
y = []
x = []
for i in cat:
    temp = [classNo] * len(testFeatureDict[i])
    y.extend(temp)
    x.extend(testFeatureDict[i])
    classNo+=1

k = []

for i in x:
    k.append(p.predict(i))


y = np.array(y)
k = np.array(k)

success = (y == k)
unique, counts = np.unique(success, return_counts=True)
print unique
print counts
