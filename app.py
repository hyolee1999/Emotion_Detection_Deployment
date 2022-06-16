from flask import Flask,render_template,Response,request
from flask_socketio import SocketIO, emit
import cv2

from tensorflow.keras.models import load_model
import numpy as np
from camera import Camera
from emotion_detection import EmotionDetection

from sys import stdout
import logging


app=Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
# app.logger.setLevel(logging.INFO)
app.config['DEBUG'] = True
socketio = SocketIO(app)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cl = {0: 'angry',1: 'disguist',2: 'fear',3: 'happy',4: 'neutral',5: 'sad',6: 'surprised'}
model = load_model('best_model.h5')
camera = Camera(EmotionDetection(model,face_cascade,cl))
# model = EmotionDetection(model,face_cascade,cl)

@socketio.on('input image', namespace='/test')
def test_message(input):

    input = input.split(",")[1]
    # app.logger.info(request.namespace.socket.sessid)
    camera.enqueue_input((request.sid,input))
    # app.logger.info(request.sid)
    image_data = camera.get_frame(request.sid)  # Do your magical Image processing here!!
    # app.logger.info(request.sid)
    # print(image_data)
    image_data = image_data.decode("utf-8")
    
    image_data = "data:image/jpeg;base64," + image_data
    # print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/test')
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    # app.logger.info("Length : %d" % len (camera.to_output))
    app.logger.info("client connected :{}".format(request.sid))

@socketio.on('disconnect', namespace='/test')
def test_connect():
    camera.to_output.pop(request.sid,None)
    # app.logger.info("Length : %d" % len (camera.to_output))
    app.logger.info("client disconnected :{}".format(request.sid))



    

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
