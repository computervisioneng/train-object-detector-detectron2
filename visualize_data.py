import os
import random

import matplotlib.pyplot as plt
import cv2


IMGS_DIR = os.path.join('.', 'data', 'train', 'imgs')
ANNS_DIR = os.path.join('.', 'data', 'train', 'anns')

if __name__ == "__main__":
    files = os.listdir(IMGS_DIR)
    while True:
        fig = plt.figure()
        k = random.randint(0, len(files) - 1)
        img = cv2.imread(os.path.join(IMGS_DIR, files[k]))
        ann_file = os.path.join(ANNS_DIR, files[k][:-4] + '.txt')

        h_img, w_img, _ = img.shape
        with open(ann_file, 'r') as f:
            lines = [l[:-1] for l in f.readlines() if len(l) > 2]
            for line in lines:
                line = line.split(' ')
                class_, x0, y0, w, h = line
                x1 = int((float(x0) - (float(w) / 2)) * w_img)
                y1 = int((float(y0) - (float(h) / 2)) * h_img)
                x2 = x1 + int(float(w) * w_img)
                y2 = y1 + int(float(h) * h_img)
                img = cv2.rectangle(img,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 255, 0),
                                    4)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
