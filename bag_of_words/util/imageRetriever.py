import os
class imageRetriever:
    def __init__(self,catToRetrieve, dir = '../vispera_loose_crop_all'):
        self.dir = dir
        self.catToRetrieve = catToRetrieve
        l = os.listdir(self.dir)

        if not set(l).issuperset(set(self.catToRetrieve)):
            raise Exception('There are classes that are not exist in catToRetrieve')


    def getDictOfDir(self, innerDir = 'train'):
        out = {}

        for i in self.catToRetrieve:
            out[i]=[]

        for i in self.catToRetrieve:
            nowDir = self.dir + '/'+ i+'/'+innerDir
            l = os.listdir(nowDir)
            for j in l:
                if '.jpg' in j:
                    out[i].append(nowDir + '/'+ j)
        return out


if __name__ == '__main__':
    nowR = imageRetriever(['bagdat1'])
    nowR.getDictOfDir()