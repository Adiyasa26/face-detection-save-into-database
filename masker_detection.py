# import the necessary packages
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from os.path import join
import numpy as np
import imutils
import cv2
import datetime 
from datetime import date
import uuid

current_time = datetime.datetime.now() 

cred = credentials.Certificate("tmb-telkom-firebase-adminsdk-nr87a-83aa727c4f.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def set_data(label):
    doc_ref = db.collection(u'mask-detection').document(uuid.uuid4().hex)
    doc_ref.set({
        u'date': [current_time.day, current_time.month, current_time.year],
        u'label': label
    })

def mask_detection_prediction(frame, faceNet, maskNet):

    # find the dimension of frame and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # create a empty list which'll store list of faces,face location and prediction
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # find the confidence or probability associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter the strong detection [confidence > min confidence(let 0.5)]
        if confidence > 0.5:

            # find starting and ending coordinates of boundry box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # make sure bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # append the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding prediction
    return (locs, preds)

# load our serialized face detector model from disk
def mask_detection_label():
    masker = []
    prototxtPath = join("deploy.prototxt")
    weightsPath = join("res10_300x300_ssd_iter_140000.caffemodel")

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("model_deteksi_masker.h5")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = mask_detection_prediction(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
                set_data(label)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                set_data(label)
        
            # include the probability in the label
            label = "{}".format(label)

            # display the label and bounding box rectangle on the output frame

            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

mask_detection_label()