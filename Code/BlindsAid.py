import cv2
import cvzone
import math
import pyttsx3
from ultralytics import YOLO

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")
# model.cuda()  # Move the model to the GPU

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#To run the Model using GPU NVIDIA Graphics Card
while True:
    success, img = cap.read()
#     img_gpu = cv2.cuda_GpuMat()  # Create a GPU matrix for the input image
# #    img_gpu.upload(img)  # Upload the input image to the GPU
#     img_gpu = cv2.cuda.resize(img_gpu, (1280, 720))  # Resize the image on the GPU
#     img = img_gpu.download()  # Download the resized image to the CPU

    results = model(img, stream=True)
    recognized_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            object_name = classNames[cls]
            recognized_objects.append(object_name)

            cvzone.putTextRect(img, f'{object_name} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Convert recognized objects into audio
    if recognized_objects:
        text = "I see " + ", ".join(recognized_objects)
        engine.say(text)
        engine.runAndWait()
