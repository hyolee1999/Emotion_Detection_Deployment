from flask import Flask,render_template,Response
from flask_socketio import SocketIO, emit
import cv2
# import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from camera import Camera
from emotion_detection import EmotionDetection
# from utils import base64_to_pil_image, pil_image_to_base64
from utils import base64_to_opencv,opencv_to_base64
from sys import stdout
import logging


app=Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['DEBUG'] = True
socketio = SocketIO(app)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cl = {0: 'angry',1: 'disguist',2: 'fear',3: 'happy',4: 'neutral',5: 'sad',6: 'surprised'}
model = load_model('best_model.h5')
# camera = Camera(EmotionDetection(model,face_cascade,cl))
model = EmotionDetection(model,face_cascade,cl)

@socketio.on('input image', namespace='/test')
def test_message(input):

    input = input.split(",")[1]
#     camera.enqueue_input(input)

#     image_data = camera.get_frame()  # Do your magical Image processing here!!
    input_img = base64_to_opencv(input)
    output_img = model.process(input_img)
    image_data = opencv_to_base64(output_img)

    image_data = image_data.decode("utf-8")
    
    image_data = "data:image/jpeg;base64," + image_data
    # print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/test')
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")

# def generate_frames():

#     while True:
            
#         ## read the camera frame
#         success,frame=camera.read()
   
#         if not success:
#             break
#         else:
       
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             faces = face_cascade.detectMultiScale(gray, 1.1,7)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
#                 image = frame[y:y + w, x:x + h,:]
#                 image = cv2.resize(image, (224, 224))


#                 image = image / 255.

#                 image = np.expand_dims(image,0)
#                 result = model.predict(image)
#                 pred = np.argmax(result)
#                 pred = cl[pred]
#                 cv2.putText(frame, pred, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#             ret,buffer=cv2.imencode('.jpg',frame)
#             frame=buffer.tobytes()

#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def generate_frames():

    app.logger.info("starting to generate frames!")
    while True:
        frame=camera.get_frame() 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    # app.run(debug=True,port=8000,host="0.0.0.0")
    socketio.run(app)
