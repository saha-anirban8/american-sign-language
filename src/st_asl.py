import json
from PIL import Image
import os

import streamlit as st
import pandas as pd
import numpy as np

from vgg16_model import VGG16Model


@st.cache()
def load_model(path: str = 'models/ASL20.pt') -> VGG16Model:
    model = VGG16Model(path_to_pretrained_model=path)
    return model


@st.cache()
def load_index_to_label_dict(
        path: str = 'src/idx_to_class.json'
        ) -> dict:
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict


@st.cache()
def load_file_structure(path: str = 'src/all_image_files.json') -> dict:
    """Retrieves JSON document outining the file structure"""
    with open(path, 'r') as f:
        return json.load(f)


def load_files(
        keys: list
        ) -> list:
    files_to_return = []
    for key in keys:
        path = os.path.join('imgs/', key)
        image_file = Image.open(path)
        files_to_return.append(image_file)
    return files_to_return



@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        alphabet: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    alphabet_dict = all_image_files.get(image_files_dtype)
    list_of_files = alphabet_dict.get(alphabet)
    return list_of_files


@st.cache()
def predict(
        img: Image.Image,
        index_to_label_dict: dict,
        model,
        k: int
        ) -> list:
    """Transforming input image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.

    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
    formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
    return formatted_predictions


if __name__ == '__main__':
    model = load_model()
    index_to_class_label_dict = load_index_to_label_dict()
    all_image_files = load_file_structure()
    types_of_letters = sorted(list(all_image_files['train'].keys()))
    types_of_letters = [letter.title() for letter in types_of_letters]


    st.title('Welcome To Project ASL!')
    instructions = """
        Select an image from the sidebar.
        The image you select will be fed
        through the VGG16 Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    selected_alphabet = st.sidebar.selectbox("Alphabet", types_of_letters)

    available_images = load_list_of_images_available(
        all_image_files, 'train', selected_alphabet)

    image_name = st.sidebar.selectbox("Image Name", available_images)

    examples_of_letters = np.random.choice(available_images, size=3)

    files_to_get = []

    for im_name in examples_of_letters:
        path = os.path.join('train', selected_alphabet, im_name)
        files_to_get.append(path)
    images_from_local = load_files(keys=files_to_get)

    img = images_from_local.pop(0)
    prediction = predict(img, index_to_class_label_dict, model, 5)

    st.title("Here is the image you've selected")
    resized_image = img.resize((336, 336))
    st.image(resized_image)
    st.title("Here are the five most likely Letters")
    df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Letter', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))

    for idx, p in enumerate(prediction):
        df.iloc[idx,
                0] = p[0]
        df.iloc[idx, 1] = p[1]
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
    st.title(f"Here are some other images of {prediction[0][0]}")

    st.image(images_from_local)
