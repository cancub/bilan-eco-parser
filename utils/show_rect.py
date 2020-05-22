import cv2
import numpy as np
import os
import pandas as pd
import sys

data_dir = '../data/'
name = sys.argv[1]
df = pd.read_pickle(os.path.join(data_dir, name + '_meta'))

def display_w_rect(sample, imdir):
    fpath = os.path.join(imdir, sample.filename)
    centre = np.array((sample.x, sample.y))
    half_dims = np.array((sample.width,sample.height))/2
    img = cv2.imread(fpath)
    cv2.circle(img, tuple(centre), 2, (0, 255, 0), 2)
    cv2.rectangle(
        img,
        tuple((centre - half_dims).astype(np.uint16)),
        tuple((centre + half_dims).astype(np.uint16)),
        (0, 255, 0),
        2
    )
    cv2.imshow("Check", img)
    if cv2.waitKey(0) == 27:
        sys.exit(0)

for i in range(len(df)):
    display_w_rect(df.sample().iloc[0], os.path.join(data_dir, name))
