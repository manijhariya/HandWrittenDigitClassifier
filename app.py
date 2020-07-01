from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import PIL.Image

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static'
configure_uploads(app, photos)

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class
'''def run_example():
    # load the image
    y = "D:/ML & AI/Final project/static/sample1.png"
    
    img = load_image(y)
    # load model
    model = load_model('final_model.h5')
    # predict the class
    digit = model.predict_classes(img)
    print(digit[0])
    return (digit)'''


@app.route('/', methods=['GET', 'POST'])
def home():
    return 'web app home page'


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'], name='sample.png')
        y = "D:/ML & AI/Final project/static/sample1.png"
        
        img = load_image(y)
        # load model
        model = load_model('final_model.h5')
        # predict the class
        digit = model.predict_classes(img)
        print(digit[0])
        #Predict_Model.load_image(filename)
        #print(filename)
        #os.rename(filename, 'sample.png')
        return render_template('upload.html',data=digit)
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=False)

