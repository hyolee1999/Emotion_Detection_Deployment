
import cv2
# import tensorflow as tf
from keras.models import load_model
import numpy as np

class EmotionDetection(object):
    def __init__(self,model,face_cascade,cl):
        self.model = model
        self.face_cascade = face_cascade
        self.cl = cl

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1,7)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            image = img[y:y + w, x:x + h,:]
            image = cv2.resize(image, (224, 224))


            image = image / 255.

            image = np.expand_dims(image,0)
            result = self.model.predict(image)
            pred = np.argmax(result)
            pred = self.cl[pred]
            cv2.putText(img, pred, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img