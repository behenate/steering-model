import os
import numpy as np
import pandas as pd
class Label:
    def __init__(self, angle, straight, detection):
        self.angle = angle
        self.straight = straight
        self.detection = detection
 
def get_labels(labels_dir):
    labels_dirs = os.listdir(labels_dir)
    labels_dirs = sorted(labels_dirs)
    labels = []
    for path in labels_dirs:
        extension = os.path.splitext(path)[1]
        if(extension == ".csv"):
            csv_file = pd.read_csv(labels_dir+path, sep=',', engine='python',error_bad_lines=False)
            rows = np.array(csv_file)
            for i, row in enumerate(rows):
                # angle row[0] sraight row[1:25]  detection row[25:49]
                label = Label(row[0], row[1:25], row[25:49])
                labels.append(label)
    return labels