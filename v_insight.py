import os
from flask import Flask, render_template, request, jsonify, Response
import cv2
import imutils
import time
import csv
from imutils import paths
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from facial_emotion_recognition import EmotionRecognition


app = Flask(__name__)

cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)


@app.route('/')

def firstpage():
    return render_template('firstpage.html')

@app.route('/next')
def index():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('secondpage.html')



@app.route('/capture', methods=['POST'])
def capture():
    Name = request.form['name']
    dataset = 'dataset'
    sub_data = Name

    path = os.path.join(dataset, sub_data)

    if not os.path.isdir(path):
        os.mkdir(path)

    info = [str(Name)]
    with open('student.csv', 'a') as csvFile:
        write = csv.writer(csvFile)
        write.writerow(info)

    print("start video stream")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)
    total = 0

    while total < 50:
        print(total)
        ret, frame = cam.read()
        img = imutils.resize(frame, width=400)
        rects = detector.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = os.path.sep.join([path, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, img)
            total += 1

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    return 'Image captured successfully'


@app.route('/preprocess', methods=['POST'])
def preprocess():
    dataset = "dataset"
    embeddingFile = "output/embeddings.pickle"
    embeddingModel = "openface.nn4.small2.v1.t7"
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    conf = 0.8

    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)

    imagePaths = list(paths.list_images(dataset))

    knownEmbeddings = []
    knownNames = []
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        print("preprocessing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:  # Fixed condition to check both width and height
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)

                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print("embedding: {}".format(total))
    data = {"embeddings": knownEmbeddings, "name": knownNames}
    f = open(embeddingFile, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("process completed")

    return jsonify({"message": "Preprocessing completed"})


@app.route('/train', methods=['POST'])
def train():
    embeddingFile = "output/embeddings.pickle"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"

    data = pickle.loads(open(embeddingFile, "rb").read())

    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["name"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(recognizerFile, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(labelEncFile, "wb")
    f.write(pickle.dumps(labelEnc))
    f.close()

    return jsonify({"message": "Training completed"})


@app.route('/recognize',methods=['POST'])
def recognize():
    def flatten(lis):
     for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

    embeddingFile = "output/embeddings.pickle"
    embeddingModel = "openface.nn4.small2.v1.t7"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"
    conf = 0.5

    print("[INFO] loading face detector.....")
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)

    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)

    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())

    box = []
    print("[INFO] start video Streaming...")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
       ret, frame = cam.read()
       frame = imutils.resize(frame, width=600)
       (h, w) = frame.shape[:2]
       imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

       detector.setInput(imageBlob)
       detection = detector.forward()

       for i in range(0, detection.shape[2]):
            confidence = detection[0, 0, i, 2]

            if confidence > conf:
               box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
               (startX, startY, endX, endY) = box.astype("int")

               face = frame[startY:endY, startX:endX]
               (fH, fW) = face.shape[:2]

               if fW < 20 or fH < 20:
                   continue

               faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
               embedder.setInput(faceBlob)
               vec = embedder.forward()

               preds = recognizer.predict_proba(vec)[0]
               j = np.argmax(preds)
               proba = preds[j]
               name = le.classes_[j]
            
            # Search for the recognized name in the CSV file
               with open('student.csv', 'r') as csvFile:
                  reader = csv.reader(csvFile)
                  name_found = False
                  for row in reader:
                        if name in row:
                           name_found = True
                           break

            # Display "unknown person" if the recognized name is not found
            if not name_found:
                name = "Unknown Person"
                proba = 0  # Set probability to 0 for unknown person

            text = "{} : {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow("frame", frame) 
            key = cv2.waitKey(1) 
            if key == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

@app.route('/emoreg', methods=['POST'])
def emotion_reg():
    er=EmotionRecognition(device='cpu')
    cam=cv2.VideoCapture(0)
    while True:
      sucess, frame = cam.read()
      frame=er.recognise_emotion(frame, return_type='BGR')
      cv2.imshow('frame',frame)
      key = cv2.waitKey(0) & 0xFF
      if key==ord('q'):
         break

    return jsonify({"message": "emotion dected click button to detect again"})
cam.release()
cv2.destroyAllWindows()


       

if __name__ == '__main__':
    app.run(debug=True)
