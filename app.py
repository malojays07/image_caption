import shutil
from charset_normalizer import detect
from numpy import object_
import streamlit as st
import io
import cv2
from  keras.preprocessing import image
from keras.applications import vgg16
import os
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
    # cap.release()
    # cap.destroyAllWindows()

def classifyObjects(video):
    splitVideo(video)
    model = vgg16.VGG16()
    classifications = []
    frames = [join('./frames', f) for f in listdir('./frames') if isfile(join('./frames', f))]
    for i in range(len(frames)):
        img = image.load_img(frames[i], target_size=(224, 224)) # load an image from file
        img = image.img_to_array(img) # convert the image pixels to a numpy array
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = vgg16.preprocess_input(img) # prepare the image for the VGG model
        img_pred = model.predict(img)
        label = vgg16.decode_predictions(img_pred)
        label = label[0][0]
        result = label[1]
        classifications.append(result)
        st.info(classifications, frames)
    return classifications, frames
        
def searchObject(searchItem, classes, frames):
    if searchItem in classes:
        index = classes.index(searchItem)
        img = frames[index]
        img = Image.open(img)
        st.image(img, caption=searchItem)
    else:
        st.write("object not found")

def app():
    if os.path.exists('./frames') :
        #os.rmdir('./frames')
        shutil.rmtree('./frames')
    else:
        os.mkdir('frames')
    st.header("Upload Video")
    st.info("Video must be less than 2MB")

    uploaded_file = st.file_uploader("video to be used in detection",type=["mp4"])

    if uploaded_file is not None:
        video = temporaryVideo(uploaded_file)
        splitVideo(video)
        classifications, frames = classifyObjects(uploaded_file)

        search_item = st.text_input('search object')
        if st.button("Search"):
            print('searching')
            searchObject(search_item, classifications, frames)


if __name__ == "__main__":
    app()