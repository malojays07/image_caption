import shutil
from numpy import object_
import streamlit as st
import io
import cv2
from keras.preprocessing import image
from keras.applications import vgg16
from os import listdir
from os.path import isfile, join
from PIL import Image

def temporaryVideo(video):
    if video is not None:
        IOBytes = io.BytesIO(video.read())
        temporary_location = ".video.mp4"
        with open(temporary_location, 'wb') as vid:
            vid.write(IOBytes.read())
        vid.close()
        return temporary_location

def splitVideo(video):
    file = temporaryVideo(video)
    cap = cv2.VideoCapture(file)
    try:
        if not os.path.exists('frames'):
            os.makedirs('frames', exist_ok=True)
    except OSError:
        print('Error: Creating directory failed')

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        path = f'./frames/{str(i)}.jpg'
        cv2.imwrite(path, frame)
        i += 1

def generate_captions(video):
    splitVideo(video)
    captions = []
    frames = [join('./frames', f) for f in listdir('./frames') if isfile(join('./frames', f))]
    model = vgg16.VGG16(weights='imagenet')
    for i in range(len(frames)):
        img = image.load_img(frames[i], target_size=(224, 224))  # load an image from file
        img = image.img_to_array(img)  # convert the image pixels to a numpy array
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = vgg16.preprocess_input(img)  # prepare the image for the VGG model
        img_pred = model.predict(img)
        label = vgg16.decode_predictions(img_pred)
        caption = label[0][0][1]  # Extract the predicted object label
        captions.append(caption)
        st.info(caption)
        st.image(frames[i], caption=caption)
    return captions, frames

def app():
    if os.path.exists('./frames'):
        shutil.rmtree('./frames')
    else:
        os.mkdir('frames')
    st.header("Upload Video")
    st.info("Video must be less than 2MB")

    uploaded_file = st.file_uploader("Video to be used in detection", type=["mp4"])
    if uploaded_file is not None and len(uploaded_file.read()) > 2e6:
        st.error("Video size exceeds the limit of 2MB. Please upload a smaller video.")
        return

    if uploaded_file is not None:
        video = temporaryVideo(uploaded_file)
        captions, frames = generate_captions(uploaded_file)

if _name_ == "_main_":
    app()
