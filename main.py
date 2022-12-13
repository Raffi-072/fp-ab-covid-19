import streamlit as st
import pandas as pd
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import pickle
from io import BytesIO


st.title('Diagnosis Covid-19')


def main():
    IMM_SIZE = 224
    lab = {'Covid': 0, 'Viral Pneumonia': 1, 'Normal': 2}
    uploaded_file = st.file_uploader("Choose a file")

    def diagnosis(file):
        # Download image
        ## YOUR CODE GOES HERE##
        image = mh.imread(file)

    # Prepare image to classification
    ## YOUR CODE GOES HERE##
        if len(image.shape) > 2:
            # resize of RGB and png images
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
        else:
            # resize of grey images
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
        if len(image.shape) > 2:
            # change of colormap of images alpha chanel delete
            image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

        # Show image
        ## YOUR CODE GOES HERE##
        # plt.imshow(image)
        # plt.show()
        plt.imshow(image)
        plt.savefig("temp.png")
        st.image("temp.png")

        # Load model
        ## YOUR CODE GOES HERE##
        from keras.models import model_from_json
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into a new model
        model.load_weights("model.h5")
        with open('history.pickle', 'rb') as f:
            history = pickle.load(f)
        print("Loaded model from disk")

        # Normalize the data
        ## YOUR CODE GOES HERE##
        image = np.array(image) / 255

        # Reshape input images
        ## YOUR CODE GOES HERE##
        image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

        # Predict the diagnosis
        ## YOUR CODE GOES HERE##
        predict_x = model.predict(image)
        predictions = np.argmax(predict_x, axis=1)
        predictions = predictions.reshape(1, -1)[0]
        print(predictions)

        # Find the name of the diagnosis
        ## YOUR CODE GOES HERE##

        diag = {i for i in lab if lab[i] == predictions}

        return diag


    if uploaded_file is not None:
        st.text(diagnosis(uploaded_file))

if __name__ == "__main__":
    main()
