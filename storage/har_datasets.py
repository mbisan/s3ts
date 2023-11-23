import os
import numpy as np

class STSDataset:

    def __init__(self,
            wsize: int = 10,
            wstride: int = 1,
            ) -> None:
        super().__init__()

        '''
            Base class for STS dataset

            Inputs:
                wsize: window size
                wstride: window stride
        '''

        self.wsize = wsize
        self.wstride = wstride

        self.splits = None

        self.STS = None
        self.SCS = None

        self.indices = None

    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = self.indices[index]-self.wsize*self.wstride
        last = self.indices[index]

        return self.STS[first:last:self.wstride,:], self.SCS[first:last:self.wstride]
    
    def sliceFromArrayOfIndices(self, indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(indexes.shape) == 1 # only accept 1-dimensional arrays

        return_sts = np.empty((indexes.shape[0], self.wsize, self.STS.shape[-1]))
        return_scs = np.empty((indexes.shape[0], self.wsize))

        for i, id in enumerate(indexes):
            ts, c = self[id]
            return_scs[i] = c
            return_sts[i] = ts

        return return_sts, return_scs
    
    def getSameClassWindowIndex(self):

        id = []
        cl = []
        for i, ix in enumerate(self.indices):
            if np.unique(self.SCS[(ix-self.wsize*self.wstride):ix]).shape[0] == 1:
                id.append(i)
                cl.append(self.SCS[ix])
        
        return np.array(id), np.array(cl)
    
    def normalizeSTS(self, mode):
        self.mean = self.STS.mean(0)
        self.percentile5 = np.percentile(self.STS, 5, axis=0)
        self.percentile95 = np.percentile(self.STS, 95, axis=0)

        self.STS = (self.STS - self.mean) / (self.percentile95 - self.percentile5)
    
class UCI_HARDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            split: str = "train",
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            UCI-HAR dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                split: "train" or "test"
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = filter(
            lambda x: "sensor.npy" in x,
            os.listdir(os.path.join(dataset_dir, "UCI HAR Dataset", split)))
        
        splits = [0]

        STS = []
        SCS = []
        for f in files:
            sensor_data = np.load(os.path.join(dataset_dir, "UCI HAR Dataset", split, f))
            STS.append(sensor_data)
            SCS.append(np.load(os.path.join(dataset_dir, "UCI HAR Dataset", split, f.replace("sensor", "class"))))

            splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS)
        self.SCS = np.concatenate(SCS)

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class HARTHDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            HARTH dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = filter(
            lambda x: ".csv" in x,
            os.listdir(os.path.join(dataset_dir, "harth")))
        
        splits = [0]

        STS = []
        SCS = []
        for f in files:
            # get separated STS
            segments = filter(
                lambda x: "acc" in x,
                os.listdir(os.path.join(dataset_dir, f[:-4])))

            for s in segments:

                sensor_data = np.load(os.path.join(dataset_dir, f[:-4], s))
                STS.append(sensor_data)
                label_data = np.load(os.path.join(dataset_dir, f[:-4], s.replace("acc", "label")))
                SCS.append(label_data)

                splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS)
        self.SCS = np.concatenate(SCS)

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

if __name__ == "__main__":
    ds = HARTHDataset("./datasets/HARTH/", wsize=64, wstride=1)