import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

class Minmax_scale:    
    def __init__(self, req):
        self.max = np.max(req)
        self.min = np.min(req)
        self.scaled = self.max - self.min
    
    def transform(self, req):
        res = (req - self.min) / self.scaled
        res[res > 1] = 1 
        res[res < 0] = 0 
        return res

class Layer:
    def __init__(self, x, n_pca):
        self.n_pca = n_pca
        self.pca = PCA(n_components=n_pca)
        x = self.pca.fit_transform(x)
        self.minmax = Minmax_scale(x)
        self.th = [(-0.01, 1.01) for i in range(n_pca)]

    def transform(self, x, mask): # x: (n_reduce_patch, n_feat), mask:(n_patch)
        z = self.pca.transform(x)
        z = self.minmax.transform(z)
        for i in range(self.n_pca):
            submask = z[:, i] < self.th[i][0]
            z[submask] = 0
            submask = z[:, i] > self.th[i][1]
            z[submask] = 0
        
        submask = mask[mask == True]

        reduce_x, reduce_z = [], []
        for i, it in enumerate(z):
            if np.sum(it) == 0:
                submask[i] = 0
            else:
                reduce_z.append(it)
                reduce_x.append(x[i])
    

        mask[mask == 1] = submask
        return np.array(reduce_x), mask, np.array(reduce_z)

    def set(self, th):
        self.th = th

class FeatureFilter:
    def __init__(self):
        self.blk = []
    
    def getFeature(self, x):
        mask = np.ones((len(x)), dtype=bool)
        for it in self.blk:
            reduce_x, mask, z = it.transform(x, mask)
            x = reduce_x
        return mask, z
    
    def addLayer(self, X, n_pca):
        if self.blk:
            mm, _ = self.getFeature(X)
            X = X[mm]
        self.blk.append(Layer(X, n_pca))

    def rmLayer(self):
        if len(self.blk) > 0:
            self.blk.pop()

    def setLayerThreshold(self, id_layer, th):
        while len(self.blk) > id_layer + 1:
            self.blk.pop()
        self.blk[id_layer].set(th)


