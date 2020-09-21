import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import collections

classNames = pickle.load( open( "firstTraining_classes.p", "rb" ))
clusterers = pickle.load( open( "firstTraining_clusters.p", "rb" ))


inquiries = ['./paris/moulinrouge/paris_moulinrouge_000119.jpg', './paris/louvre/paris_louvre_000111.jpg',
             './paris/eiffel/paris_eiffel_000105.jpg','./paris/eiffel/paris_eiffel_000107.jpg']

inq = 1
#PREDICT
queryImageName = inquiries[inq]
queryImage = cv2.imread(queryImageName)
if(type(queryImage) != np.ndarray):
    raise ValueError('The image could not be read')

nowImageGray= cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(nowImageGray, None)
siftOut = sift.compute(nowImageGray, kp)
inquiryVectors = siftOut[1]

allVotes = []
for v in inquiryVectors:
    minDistance =  10**6
    nowVote = 0
    for i in range(len(clusterers)):
        temp = v.reshape(1,-1)
        predictedCluster = clusterers[i].predict(temp)
        nowDistance = np.linalg.norm(clusterers[i].cluster_centers_[predictedCluster]- v)
        if(minDistance>nowDistance):
            minDistance = nowDistance
            nowVote = i
    allVotes.append(nowVote)


print allVotes
print collections.Counter(allVotes)

print 'out guess is ', classNames[max(set(allVotes), key=allVotes.count)]

