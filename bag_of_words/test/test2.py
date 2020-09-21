import pickle

from bagOfWord.util.extractor import *

trainingName = 'fall_svm'

svmOutDict = pickle.load( open( trainingName, "rb" ))
clf = svmOutDict['svm']
cat = svmOutDict['cat']
numOfWords = svmOutDict['numOfWords']
maxVectorNumberFromImage = svmOutDict['maxVecFromImage']
kmeans = svmOutDict['kmeans']



testExtractor = featureExtractor(cat,maxVectorNumberFromImage,numOfWords,False, kmeans)
testFeatureDict = testExtractor.getFeatures()

classNo = 0.0
y = []
x = []
for i in cat:
    temp = [classNo] * len(testFeatureDict[i])
    y.extend(temp)
    x.extend(testFeatureDict[i])
    classNo+=1


k = clf.predict(x)
y = np.array(y)
k = np.array(k)

success = (y == k)
unique, counts = np.unique(success, return_counts=True)
print unique
print counts
