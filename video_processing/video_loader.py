import os
import cv2
import glob
from tqdm import tqdm

def load_video(inp: str, out: str, lim = -1):
    vidcap = cv2.VideoCapture(inp)
    cfg = {}
    cfg['fps'] = int(vidcap.get(cv2.CAP_PROP_FPS))
    cfg['frame_count'] = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {inp} ({cfg['fps']} fps)")
    print(f"Extracting frames to {out}")
    li = glob.glob(out + '*.jpg')
    if len(li) == cfg['frame_count']:
        print("already prepare images")
        return cfg
    
    success, image = vidcap.read()
    for count in tqdm(range(cfg['frame_count'])):
        if success:
            cv2.imwrite(
                os.path.join(out, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1
        if count == lim:
            break
        
    cfg['count'] = count
    return cfg



