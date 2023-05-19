# import the necessary packages
from flask import Flask, render_template, Response, request
from camera2 import VideoStream
import time
import threading
import os

pi_camera = VideoStream(resolution=(1280,720),framerate=30) # flip pi camera if upside down.

# App Globals
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #editable index.html

def gen():
    #get camera frame
    while True:
        frame = pi_camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
    




