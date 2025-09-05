import cv2
from matplotlib import pyplot as plt
import numpy as np
import ultralytics
from ultralytics import YOLO
import torch
from ultralytics.utils.plotting import Annotator, colors

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
test01 = "/home/blitzkrieg/source/repos/Workshop/Local/data/dataset/elektrik/Pole2_Yolo/images/train/16_3450.png"
test02 = "/home/blitzkrieg/source/repos/Workshop/Local/data/dataset/elektrik/Pole2_Yolo/images/test/16_3735.png"
test03 = "data/DJI_20240905091530_0003_W.JPG"
test_image = test03

LABELS = {0: "Boş", 1: "Çelik Direkler", 2: "Kafes Kule", 3: "Kablo", 4: "Ahşap Kule"}
colorMap = {"Boş":"#ffffff", "Çelik Direkler":"#0000ff", "Kafes Kule":"#ff0000", "Kablo":"#00ff00", "Ahşap Kule":"#ff0000"}

# Load a model
model = YOLO("Weight/yolov9c-cable-seg.pt")  # load a custom model
model.fuse()


def ParseResults(results, threshold=0.5, scale_masks=True):
    batches = []
    
    SCORES = torch.Tensor([]).to(DEVICE)
    CLASSES = torch.Tensor([]).to(DEVICE)
    MASKS = torch.Tensor([]).to(DEVICE)
    BOXES = torch.Tensor([]).to(DEVICE)

    with torch.no_grad():
        for result in results:
            original_shape = result.orig_shape
            _scores = result.boxes.conf        # 7
            _classes = result.boxes.cls         # 7
            _masks = result.masks.data         # 7, 480, 640
            _boxes = result.boxes.xyxy         # 7, 4
            
            # Threshold Filter
            conditions = _scores > threshold
            SCORES = torch.cat((SCORES, _scores[conditions]), dim=0)
            CLASSES = torch.cat((CLASSES, _classes[conditions]), dim=0)
            BOXES = torch.cat((BOXES, _boxes[conditions]), dim=0)
            mask = _masks[conditions]
            if scale_masks:
                mask = ScaleMasks(mask, original_shape[:2])

            MASKS = torch.cat((MASKS, mask), dim=0)
            
            batches += [(SCORES, CLASSES, MASKS, BOXES)]

    return batches
 

def ScaleMasks(masks: torch.Tensor, shape: tuple) -> torch.Tensor:
    masks = masks.unsqueeze(0)
    interpolatedMask:torch.Tensor = torch.nn.functional.interpolate(masks, shape, mode="nearest")
    interpolatedMask = interpolatedMask.squeeze(0)
    return interpolatedMask


def DrawResults(image, scores: torch.Tensor, classes: torch.Tensor, masks: torch.Tensor, boxes: torch.Tensor, labels:dict=LABELS, class_filter:list=None):
    _image = np.array(image).copy()
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    maskCanvas = np.zeros_like(_image)


    with torch.no_grad():
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy().astype(np.int32)
        masks = masks.cpu().numpy()
        boxes = boxes.cpu().numpy()
    
    for score, cls, mask, box in zip(scores, classes, masks, boxes):
        label = labels[cls]

        if class_filter and cls not in class_filter:
            continue

        box = box.astype(np.int32)
        mask = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        maskCanvas = cv2.addWeighted(maskCanvas, 1.0, mask, 1.0, 0)
        maskCanvas = cv2.rectangle(maskCanvas, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=3)  # Red color for bounding box
        maskCanvas = cv2.putText(maskCanvas, f"{label} : {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=(255, 0, 0), thickness=2)
    
    canvas = cv2.addWeighted(_image, 1.0, maskCanvas.astype(np.uint8), 1.0, 0)
    return canvas, maskCanvas


def RescaleTheMask(orijinal_image, masks):
    _masks = []
    for contour in masks:
        b_mask = np.zeros(orijinal_image.shape[:2], np.uint8)
        contour = contour.astype(np.int32)
        # contour = contour.reshape(-1, 1, 2)

        w = orijinal_image.shape[0]
        h = orijinal_image.shape[1]

        mask = cv2.drawContours(b_mask, [contour], -1, (1, 1, 1), cv2.FILLED)
        _masks += [mask]

    return _masks



image = cv2.imread(test_image)


with torch.no_grad():
    results = model(
        image,
        save=False,
        show_boxes=False,
        project="./inference/",
        conf=0.5,
        iou=0.7,
        retina_masks=False
    )

    batches = ParseResults(results, threshold=0.5, scale_masks=True)
    scores, classes, masks, boxes = batches[0]

    canvas, mask = DrawResults(image, scores, classes, masks, boxes, class_filter=[3])


    # ALL Segmentation
    # canvas = torch.any(result.masks.data, dim=0).int() * 255
    
    # Instance Segmentation
    # objIdx = torch.where(result.boxes.cls.data == 3)
    # objMasks = result.masks.data[objIdx]
    # obj_mask = torch.any(objMasks, dim=0).int() * 255
    


#! Plot
fig, axs = plt.subplots(2, 2, figsize=(27, 15))
axs[0][0].imshow(image)
axs[0][0].set_title("Orijinal Görüntü")

axs[0][1].imshow(mask)
axs[0][1].set_title("Segmentasyon Maskesi")

# axs[1][0].imshow(obj_mask.cpu().numpy())
# axs[1][0].set_title("Seçilen")

axs[1][1].imshow(canvas)
axs[1][1].set_title("Sonuç")

# mask = np.array(obj_mask.cpu().numpy())*255
# cv2.imwrite("cable_mask.png", mask)

plt.tight_layout()
plt.show()
