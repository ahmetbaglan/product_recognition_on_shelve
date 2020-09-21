import pickle

from bagOfWord.util.extractor import *
# import os, sys
# lib_path = os.path.abspath(os.path.join('..','libsvm-3.21','python'))
# sys.path.append(lib_path)
# from svmutil import *

from sklearn import svm

trainingName = 'fall'
cat = ['bagdat1', 'biskrem1', 'burn1']
maxVectorNumberFromImage= 10000
numOfWords= 250
e = featureExtractor(cat,maxVectorNumberFromImage,numOfWords)
trainFeaturesDict = e.getFeatures()

classNo = 0.0
y = []
x = []
for i in cat:
    temp = [classNo] * len(trainFeaturesDict[i])
    y.extend(temp)
    x.extend(trainFeaturesDict[i])
    classNo+=1

# prob  = svm_problem(y, x, isKernel=True)
# param = svm_parameter('-t 0 -c 4 -b 1')
# m = svm_train(prob, param)
# p_labels, p_acc, p_vals  = svm_predict(y, x, m, '-b 1')

clf = svm.SVC()
clf.fit(x,y)

kmeans = e.getKmeans()
centers = e.getClassCenterFeatures()


svmOutDict = {}
svmOutDict['svm'] = clf
svmOutDict['cat'] = cat
svmOutDict['numOfWords'] =  numOfWords
svmOutDict['maxVecFromImage'] = maxVectorNumberFromImage
svmOutDict['kmeans'] = kmeans

pickle.dump(svmOutDict, open( trainingName+'_svm', "wb" ))

correlationOutDict = {}
correlationOutDict['centers'] = centers
correlationOutDict['numOfWords'] = numOfWords
correlationOutDict['cat'] = cat
correlationOutDict['maxVecFromImage'] = maxVectorNumberFromImage
correlationOutDict['kmeans'] = kmeans

pickle.dump(correlationOutDict, open( trainingName+'_correlation', "wb" ))


########################################################################################################




