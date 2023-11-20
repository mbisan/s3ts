from typing import Callable, Generator
from collections import deque
import numpy as np

class StreamSimulator(Generator):

    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 patts: np.ndarray,
                 wdw_len: int, 
                 wdw_str: int,
                 infinite_STS: Generator,
                 image_method: Callable,
                 image_args: dict = None,
                 discard: int = 100
                 ) -> None:

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())

        self.X, self.Y, self.patts = X, Y, patts
        self.wdw_len, self.wdw_str= wdw_len, wdw_str 
        self.STS_gen = infinite_STS(X, Y)
        self.image_method, self.image_args = image_method, image_args
        maxlen = wdw_len*wdw_str
        self.STS, self.SCS = deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.DM = deque(maxlen=maxlen)

        self.prev_col = None
        for _ in range(discard+maxlen):
            xi, yi = next(self.STS_gen)
            xi = np.expand_dims(xi, 1)
            self.STS.append(xi), self.SCS.append(yi)
            dmc = image_method(STS=xi, patts=self.patts,
                prev_col=self.prev_col, **self.image_args)
            self.DM.append(dmc)
            self.prev_col = np.squeeze(dmc, 2)

    def send(self, _):
        xi, yi = next(self.STS_gen)
        xi = np.expand_dims(xi, 1)
        self.STS.append(xi), self.SCS.append(yi)
        dmc = self.image_method(STS=xi, patts=self.patts,
                prev_col=self.prev_col, **self.image_args)
        self.DM.append(dmc)
        self.prev_col = np.squeeze(dmc, 2)

        idx = self.wdw_len*self.wdw_str-1

        pxi = np.hstack(self.STS)[:,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        pyi = np.array(self.SCS)[0]
        pdm = np.squeeze(np.swapaxes(
            np.array(self.DM),0,3))[:,:,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]

        return pxi, pyi, pdm

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def samples_from_simulator(
        sim: StreamSimulator,          
        nsamp: int,
        mode: str = "det",      #["det", "prob"]
        every_n: int = 10,
        acc_prob: float = 0.1,
        seed: int = 42,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    series = []
    labels = []
    frames = []

    if mode == "det":
        for _ in range(nsamp):
            for _ in range(every_n):
                si, li, fi = next(sim)
            series.append(si)
            labels.append(li)
            frames.append(fi)
    if mode == "prob":
        rng = np.random.default_rng(seed=seed)
        while len(labels) < nsamp :
            si, li, fi = next(sim)
            if rng.uniform() < acc_prob: 
                series.append(si)
                labels.append(li)
                frames.append(fi)
                nsamp += 1

    return np.array(series), np.array(labels), np.array(frames)
