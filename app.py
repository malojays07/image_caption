import shutil
import streamlit as st
import cv2
from keras.preprocessing import image
from keras.applications import vgg16
from os import listdir
from os.path import isfile, join
from PIL import Image
import tempfile

def splitVideo(video):
    frames_dir = tempfile.mkdtemp()  # Create a temporary directory to store frames
    cap = cv2.VideoCapture(video)
    try:
        if not os.path.exists('frames'):
            os.makedirs('frames', exist_ok=True)
    except OSError:
        print('Error: Creating directory failed')
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        path = f'{frames_dir}/{str(i)}.jpg'
        cv2.imwrite(path, frame)
        i += 1
    return frames_dir

def generate_captions(video):
    frames_dir = splitVideo(video)
    captions = []
    frames = [join(frames_dir, f) for f in listdir(frames_dir) if isfile(join(frames_dir, f))]
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
    shutil.rmtree(frames_dir)  # Remove the temporary frames directory
    return captions, frames

def app():
    st.header("Upload Video")
    st.info("Video must be less than 2MB")

    uploaded_file = st.file_uploader("Video to be used in detection", type=["mp4"])
    if uploaded_file is not None and len(uploaded_file.read()) > 2e6:
        st.error("Video size exceeds the limit of 2MB. Please upload a smaller video.")
        return

    if uploaded_file is not None:
        captions, frames = generate_captions(uploaded_file)

if __name__ == "__main__":
    app()
