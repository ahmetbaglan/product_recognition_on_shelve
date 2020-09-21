import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle

#Get all the image adresses in a list

trainingName = 'firstTraining'
numOfImageFromEachClass = 95
maxVectorNumberFromImage = 100
numOfCluster = 20


categories = ['eiffel','louvre','defense','moulinrouge','pantheon','sacrecoeur']

class_List = []
for i in categories:
    class_List.append('./paris/'+i+'/paris_'+i+'_0000')

allVectors = {}
classNames = -1

for c in class_List:
    classNames += 1
    allVectors[classNames]=[]
    for j in range(numOfImageFromEachClass):
        nowImageName = c + str(j).zfill(2) + '.jpg'
        try:
            nowImage = cv2.imread(nowImageName)
            if(type(nowImage) != np.ndarray):
                raise ValueError('The image could not be read')

        except:
            continue

        nowImageGray = cv2.cvtColor(nowImage, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp = sift.detect(nowImageGray, None)
        siftOut = sift.compute(nowImageGray, kp)
        imageVectors = siftOut[1]
        till = min(maxVectorNumberFromImage, len(imageVectors))
        np.random.shuffle(imageVectors)
        my_sift_vectors = imageVectors[0:till]
        allVectors[classNames].extend(my_sift_vectors)

clustererList = []

for i in range(len(class_List)):
    a = np.asarray(allVectors[i])
    kmeans = KMeans(n_clusters=numOfCluster, random_state=0).fit(a)
    clustererList.append(kmeans)

pickle.dump(clustererList, open( trainingName+'_clusters.p', "wb" ))
pickle.dump(categories, open( trainingName+'_classes.p', "wb" ))
