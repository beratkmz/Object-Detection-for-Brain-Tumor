# Object Detection for Brain Tumor with OpenCV and YOLO on Google Colab

<p align="center">
  <img src="https://github.com/user-attachments/assets/3dbed111-e15d-4f17-964a-033553fb3713" width="400" />
  <img src="https://github.com/user-attachments/assets/d235151d-d991-4549-b4aa-d318eb3ddc6c" width="400" />
</p>


![results (1)](https://github.com/user-attachments/assets/ced30f56-422a-4ef5-8a4b-79b15f36c832)



This project demonstrates how to build OpenCV using Google Colab, how to use it for real-time object detection using the YOLO model (`yolov8n.pt`).

## Requirements
- OpenCV into Google Colab
- YOLOv11 or another compatible YOLO model
- Python libraries: `opencv-python`, `os`, `imutils`, and `ultralytics`

## Features
- Fast training with Google Colab environment.
- Real-time object detection using the YOLO model
- Python-based implementation for ease of use and customization
- Dockerized environment for easy deployment across different systems

## Installation

Follow the steps below to build OpenCV into Google Colab, set up the YOLO model for object detection.

### Step 1: Creating a file into the Google Drive

First, create a folder in Google Drive and upload all the attached files into it. Datasets folder will be empty. I cannot share datasets due to copyright. You can also use this layout for different projects. <br> 
<br>
My open source dataset: https://universe.roboflow.com/gadjah-mada-university/brain-tumor-detection-tcdk4
<br>

### Step 2: Creating .ipynb Google Colab file for coding
Add a Google Colab file to Google Drive to store your all process.


### Step 3: Connect Google Colab with Google Drive and environment preparation
In this project Google Colab will be used as computer and Google Drive will be used as storage, for this reason we need to integrate these two applications together.
In Google Colab file ...... .ipynb

```sh
from google.colab import drive
drive.mount("/content/drive")
```
```sh
%pwd
%cd /content/drive/MyDrive/YOLOv8/brain_tumor_detection
```
### Step 4: Installation of Ultralytics

```sh
%pip install ultralytics

import ultralytics
ultralytics.checks()
```

### Step 5: Traning Model
Firstly the data .zip file need to unzip.
```sh
!unzip data/brain_tumor_dataset.zip -d ./data
```

After that we can train our model with `yolov8n.pt` with using config.yaml file.
```sh
!yolo detect train data=data/config.yaml model=yolov8n.pt epochs=25 imgsz=640 workers=8 batch=8 device=0 name=yolov8_brain_tumor_detection
```
If the training is interrupted, continue with this code
```sh
!yolo detect train model=runs/detect/yolov8_brain_tumor_detection/weights/last.pt resume=True
```

### Step 6: Tumor Detection / Prediction
Option 1: Prediction with YOLO's ready code
```sh
!yolo detect predict model=runs/detect/yolov8_brain_tumor_detection/weights/best.pt source=inference save=True
```
Option 2: Prediction with Python code
```sh
import cv2
import imutils
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

font = cv2.FONT_HERSHEY_SIMPLEX
img_path = "inference/128.jpg"
model_path ="runs/detect/yolov8_brain_tumor_detection/weights/best.pt"

img = cv2.imread(img_path)
img = imutils.resize(img, width=360)
model = YOLO(model_path)

results = model(img)[0]

threshold = 0.5
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    # print(result)
    if score > threshold:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        class_name = results.names[int(class_id)]
        score = score * 100
        text = f"{class_name}: %{score:.2f}"
        cv2.putText(img, text, (int(x1), int(y1)-10), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
cv2_imshow(img)
```
Take all the images in the inference into a for loop and print them all out
```sh
import os
import cv2
import imutils
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Model and font settings
font = cv2.FONT_HERSHEY_SIMPLEX
model_path = "runs/detect/yolov8_brain_tumor_detection/weights/best.pt"
model = YOLO(model_path)
threshold = 0.5

# Loop all images in inference folder
for filename in os.listdir("inference"):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Filter image files
        img_path = os.path.join("inference", filename)
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=360) 

        # Make object detection
        results = model(img)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                # Kutuyu ve etiketi çiz
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                class_name = results.names[int(class_id)]
                score = score * 100
                text = f"{class_name}: %{score:.2f}"
                cv2.putText(img, text, (int(x1), int(y1) - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2_imshow(img)
        print(f"İşlenen resim: {filename}")
```
## License
MIT
