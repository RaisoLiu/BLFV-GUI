import os
import cv2
import glob
import random
from video_processing import load_video
from matplotlib import pyplot as plt

class VideoManager:
    def __init__(self, name, export, kernel_frame_size=16):
        self.export = export
        self.frame_dir = self.export  + 'frame/'
        os.makedirs(self.export, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)
        
        self.video_cfg = load_video(name, self.frame_dir)
        self.full_frame = sorted(glob.glob(self.frame_dir + '*.jpg'))
        self.n_frame = len(self.full_frame)
        self.kernel_frame = random.choices(self.full_frame, k=kernel_frame_size)
    
    def getframe(self, index):
        return cv2.imread(self.full_frame[index])
    
    def getkernelframe(self):
        res = []
        for it in self.kernel_frame:
            frame = cv2.imread(it)
            res.append(frame)
        return res