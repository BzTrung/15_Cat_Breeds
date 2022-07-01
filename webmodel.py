from flask import Flask, render_template, request, Response
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import  sys
from keras.models import load_model
from adabelief_tf import AdaBeliefOptimizer
from io import BytesIO

url = 'in4.csv'
data = pd.read_csv('in4.csv', encoding='utf-8')
target_names = data['ten']
features_of_cat = data['dactinh']


# Load model
model = load_model("val_accuracy-0.75.h5")

class VideoCamera(object):

    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
       #extracting frames

        ret, image = self.video.read()
        image = cv2.flip(image,1)
        image_pred = cv2.resize(image, (224, 224))
        image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
        image_pred=image_pred.reshape(1,224,224,3)
        image_pred=image_pred/255.0
        pred = model.predict(image_pred)
        pred_name = target_names[np.argmax(pred)]
        cv2.putText(image ,pred_name,(30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,0,0),2,cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


# Flask Init
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/img_upload"
app.config['Save_chart'] = "static/chart"

@app.route('/')
def home():
        return render_template("index.html")

@app.route("/", methods=['GET', 'POST'])
def img_pred():
    if request.method == "POST":
         try:
            image = request.files['file']
            if image:
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(path_to_save)
                img=cv2.imread(path_to_save)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(224,224))
                img=img.reshape(1,224,224,3)
                img=img/255.0
                pred = model.predict(img)
                plt.subplots(figsize=(10,5))
                plt.barh(y=target_names, width=pred[-1,]*100,color = "red")
                path_chart_pred = os.path.join(app.config['Save_chart'], "pred_chart")
                plt.savefig(path_chart_pred)
                num_cat =  np.argmax(pred)
                return render_template("index.html",idb = True,user_image = image.filename,
                                                    msg="Dự đoán " + str("%.2f" % (pred[-1,num_cat]*100))
                                                    +"% là giống mèo " + target_names[num_cat],
                                                    pred_name = target_names[num_cat],
                                                    extra=features_of_cat[num_cat])

         except Exception as ex:
            return render_template('index.html')

    else:
        return render_template('index.html')




@app.route("/webcam_classify", methods=['GET', 'POST'])
def webcam_classify():
    return render_template('webcam_classify.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')        
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = os.environ.get("PORT",5000)
    app.run(host='0.0.0.0', debug=False,port=port)