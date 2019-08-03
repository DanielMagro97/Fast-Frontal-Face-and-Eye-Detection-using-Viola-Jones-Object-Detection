import cv2
import os
import pandas as pd

# indicate whether positive or negative images are being shown
eval_pos_or_neg = "pos"

# create a dataframe to store the results of evaluation
index = ["pre-trained_default", "pre-trained_alt",
         "pos150k_neg75k_stages8", "pos150k_neg75k_stages10", "pos150k_neg75k_stages12", "pos150k_neg75k_stages15",
         "pos75k_neg150k_stages8", "pos75k_neg150k_stages10", "pos75k_neg150k_stages12", "pos75k_neg150k_stages15",
         "pos150k_neg150k_stages10",
         "pos50k_neg100k_stages10", "pos25k_neg50k_stages10"]
columns = ["true positives", "true negatives", "false positives", "false negatives", "precision", "recall", "FMeasure"]
results = pd.DataFrame(index=index, columns=columns)
results = results.fillna(0)

# directory of the evaluation images
eval_imgs_directory = "evaluation"

# loading the pre-trained cascades
pretrained_cascade_default = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
pretrained_cascade_alt = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
# loading the cascades I trained
pos150k_neg75k_stages8  = cv2.CascadeClassifier('mycascades/cascade_pos150k_neg75k_stages8.xml')
pos150k_neg75k_stages10 = cv2.CascadeClassifier('mycascades/cascade_pos150k_neg75k_stages10.xml')
pos150k_neg75k_stages12 = cv2.CascadeClassifier('mycascades/cascade_pos150k_neg75k_stages12.xml')
pos150k_neg75k_stages15 = cv2.CascadeClassifier('mycascades/cascade_pos150k_neg75k_stages15.xml')

pos75k_neg150k_stages8  = cv2.CascadeClassifier('mycascades/cascade_pos75k_neg150k_stages8.xml')
pos75k_neg150k_stages10 = cv2.CascadeClassifier('mycascades/cascade_pos75k_neg150k_stages10.xml')
pos75k_neg150k_stages12 = cv2.CascadeClassifier('mycascades/cascade_pos75k_neg150k_stages12.xml')
pos75k_neg150k_stages15 = cv2.CascadeClassifier('mycascades/cascade_pos75k_neg150k_stages15.xml')

pos150k_neg150k_stages10 = cv2.CascadeClassifier('mycascades/cascade_pos150k_neg150k_stages10.xml')

pos50k_neg100k_stages10 = cv2.CascadeClassifier('mycascades/cascade_pos50k_neg100k_stages10.xml')
pos25k_neg50k_stages10 = cv2.CascadeClassifier('mycascades/cascade_pos25k_neg50k_stages10.xml')

cascades = [pretrained_cascade_default, pretrained_cascade_alt,
            pos150k_neg75k_stages8, pos150k_neg75k_stages10, pos150k_neg75k_stages12, pos150k_neg75k_stages15,
            pos75k_neg150k_stages8, pos75k_neg150k_stages10, pos75k_neg150k_stages12, pos75k_neg150k_stages15,
            pos150k_neg150k_stages10,
            pos50k_neg100k_stages10, pos25k_neg50k_stages10]

# iterate over every image in the evaluation image directory
for eval_img_path in os.listdir(eval_imgs_directory):
    # load the image from disk in grayscale
    img = cv2.imread(eval_imgs_directory+"\\"+eval_img_path, cv2.IMREAD_GRAYSCALE)

    # iterate through all the cascades and calculate how many faces are detected in each image
    for cascade_name, cascade in zip(index, cascades):
        # detect the number of faces in the image using the current cascade.
        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if eval_pos_or_neg == "pos":
            if len(faces) >= 1:
                results.at[cascade_name,"true positives"] += 1
                # if more than 1 face is detected, add the extra detections as false positives
                results.at[cascade_name, "false positives"] += len(faces) - 1
            if len(faces) < 1:
                results.at[cascade_name, "false negatives"] += 1

        elif eval_pos_or_neg == "neg":
            if len(faces) >= 1:
                results.at[cascade_name, "false positives"] += 1
            if len(faces) < 1:
                results.at[cascade_name, "true negatives"] += 1

# Calculate Precision, Recall
results["precision"] = results["true positives"] / (results["true positives"] + results["false positives"])
results["recall"] = results["true positives"] / (results["true positives"] + results["false negatives"])
results["FMeasure"] = 2 * (results["precision"]*results["recall"]) / (results["precision"]+results["recall"])

print(results.to_string())
