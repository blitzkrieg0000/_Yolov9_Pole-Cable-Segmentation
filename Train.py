# #! Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
import random

import torch
torch.cuda.empty_cache()
from Tool.Core import CalculateConfusionMatrix
from ultralytics import YOLO

random_color = lambda: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def PrepareModel():
    model = YOLO("Model/yolov9c-seg.yml")
    model = model.load("Weight/yolov9c-seg.pt")
    return model


def TrainModel(model, config):
    results = model.train(**config)
    return results


if "__main__" == __name__:
    # Argümanların override sırası: args >  "cfg" > "defaults.yml"
    config = {
        "data" : "/home/blitzkrieg/source/repos/Workshop/Local/data/dataset/elektrik/Pole2_Yolo/pole2_yolo.yml",
        "epochs" : 300,
        "imgsz" : 640,
        "batch" : 8,
        "project" : "results",
        "name" : "cable_segmentation_train",
        "verbose" : True,
        "plots" : True,
        "save" : True,
        "cfg" : "cfg/cable_segmentation_train.yml"
    }

    # Load Model
    model = PrepareModel()
    
    # Train Model
    results = TrainModel(model, config)
    
    # Confusion Matrix
    confusion_matrix = results.confusion_matrix.matrix
    labels = model.names
    
    RESULTS = CalculateConfusionMatrix(confusion_matrix, transpoze=True, labels=labels)




