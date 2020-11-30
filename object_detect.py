
# this program will populate the core_data.csv file with results of object detection on respective image

import cv2
import os
import sys
import pandas as pd
import numpy as np

# Root directory of the project
MAIN_ROOT_DIR = os.path.abspath(".")
print(MAIN_ROOT_DIR)
#Path for root directory Mask RCNN folder
ROOT_DIR = os.path.join(MAIN_ROOT_DIR, "Mask_RCNN")
print(ROOT_DIR)

# Import Mask RCNN local libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

#This is a python file created for training the MaskRCNN model for sepcific application
# it contains details/configuration specific to application here pothole
import pot_hole

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
POTHOLE_MODEL_PATH = os.path.join(MAIN_ROOT_DIR, "mask_rcnn_pot_hole_0007.h5")
print(POTHOLE_MODEL_PATH)

# Directory of images to run detection on
#Note main_dir is same as MAIN_ROOT_DIR
main_dir = os.path.abspath(".")
IMAGE_DIR =  os.path.join(main_dir, "road_images")# r"C:\Users\manis\PycharmProjects\pothole_heatmap\road_images"
print("img_dir", IMAGE_DIR)

"""
## Configurations

We'll be using a model trained on the pothole dataset. The configurations of this model are in the ```Pot_holeConfig``` class in ```pot_hole.py```.

For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```Pot_holeConfig``` class and override the attributes you need to change.
"""
config = pot_hole.Pot_holeConfig()

class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


"""
## Create Model and Load Trained Weights
"""
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights of our model
model.load_weights(POTHOLE_MODEL_PATH, by_name=True)


"""
## Class Names

The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.

To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.

To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.

"""

# Here we are declearing the names of our classes
class_names = ['BG', 'Pot_hole']




#This function defines differentcolur for each type of object during object detection
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

#Function places coloured tanslucent mask on the pixels where object is identified
def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

#This function add boxes to object, calls the apply mask function, and return the final image
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
#This function store the object detection values in the columnsof new dataframe against each image
def save_results(df, img, boxes, masks,classid, names, scores):
    print(img)
    df["boxes"][img] = boxes
    print(df["boxes"][img])
    #Below we have commented out the the string of mask value in csv,
    #this is because, it is of dimention 1024*1024 hence it is not stored in usable form
    #In future wi will convert such that it becomes usable
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

#creating list of al the image for inference
images = os.listdir(IMAGE_DIR)
print(images)

#loading core _data csv file to populate it furthe with object detection results, according added new Columns
core_data_loc = os.path.join(main_dir, "core_data.csv")
df= pd.read_csv(core_data_loc)
df = df.set_index("Image")
df["boxes"] = 1
df["masks"] = 1
df["class_ids"] = 1
df["class_names"] = 1
df["scores"] = 1
print(df)

#The Results of object detection are multidimentional hence defininf the dtaatype for each columens
convert_dict = {"boxes": object, "masks": object, "class_ids": object, "class_names": object,"scores":object,}
df = df.astype(convert_dict)
print(df)

for img in images:
    print(img)

    #reading each image
    image = cv2.imread(os.path.join(IMAGE_DIR, img))

    #Converting it to a size that is compatible to our model
    image= cv2.resize(image, (1024, 1024))

    #Runnig Inference
    results = model.detect([image], verbose=1)
    r = results[0]

    #Printing the object detection results just for reference
    print(type(r),type(r['rois']),type(r['masks']),type(r['class_ids']),type(class_names), type(r['scores']))
    masked_image= display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    loc = os.path.join(main_dir, "pot_holes_detected",  img)
    print(loc)

    #Saving the masked image into pothole_detected folder
    cv2.imwrite(loc,masked_image)

    #Saving the results of object detection into the dataframe
    all_data = save_results(df,img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    print(all_data)

    #Coverting dataframe into CSV
    super_core_data_loc = os.path.join(main_dir, "super_core_data.csv")
    print(super_core_data_loc)
    all_data.to_csv(super_core_data_loc)

# closing all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
