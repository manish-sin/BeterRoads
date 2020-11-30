# Step 3: run demon.py
# this program will populate the core_data.csv file with results of object detection on respective image

import cv2


# %%
import os
import sys
import random
from PIL import Image
import math
import numpy as np

import skimage.io

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
#from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

import pot_hole


#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_pot_hole_0007.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)
print(COCO_MODEL_PATH)
# Directory of images to run detection on
main_dir = os.path.abspath("../../")
IMAGE_DIR =  os.path.join(main_dir, "road_images")# r"C:\Users\manis\PycharmProjects\pothole_heatmap\road_images"
print("img_dir", IMAGE_DIR)
# %%
"""
## Configurations

We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.

For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.
"""
config = pot_hole.Pot_holeConfig()
# %%
class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# %%
"""
## Create Model and Load Trained Weights
"""

# %%
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# %%
"""
## Class Names

The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.

To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.

To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
```
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
```

We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)
"""

# %%
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Pot_hole']

# %%

##Visualisation Function

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

def save_results(df, img, boxes, masks,classid, names, scores):
    print(img)
    df["boxes"][img] = boxes
    print(df["boxes"][img])
    #df["masks"][img] = masks
    #print(df["masks"][img])
    df["class_ids"][img] = classid
    print(df["class_ids"][img])
    df["class_names"][img] = names
    #print(df["class_names"][img])
    df["scores"][img] = scores
    print(df["scores"][img])
    return df
"""
## Run Object Detection
"""

# # %%
# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
images = os.listdir(IMAGE_DIR)
print(images)

import pandas as pd
import numpy as np
core_data_loc = os.path.join(main_dir, "core_data.csv")
df= pd.read_csv(core_data_loc)
df = df.set_index("Image")
df["boxes"] = 1
df["masks"] = 1
df["class_ids"] = 1
df["class_names"] = 1
df["scores"] = 1

convert_dict = {"boxes": object, "masks": object, "class_ids": object, "class_names": object,"scores":object,}
df = df.astype(convert_dict)
print(df)
for img in images:
    print(img)
    image = cv2.imread(os.path.join(IMAGE_DIR, img))
    image= cv2.resize(image, (1024, 1024))
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    #print(r, r['class_ids'])
    print(type(r),type(r['rois']),type(r['masks']),type(r['class_ids']),type(class_names), type(r['scores']))

    masked_image= display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    loc = os.path.join(main_dir, "pot_holes_detected",  img)
    print(loc)
    #loc = r"pot_holes_detected\%s" %img
    cv2.imwrite(loc,masked_image)
    all_data = save_results(df,img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    all_data.to_csv("super_core_data.csv")
    #cv2.imwrite(file_names, masked_image)
    # cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
    # cv2.resize(masked_image, (600,600))
    # cv2.imshow("Masked Image", masked_image)
    # %%

cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
