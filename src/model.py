import keras
import numpy as np

#PARAMS
dataset = './dataset/preprocessed_data.json'
epochs = 10
batch_size = 32

locations = ['home', 'work', 'public']
times = ['wd-evening','wd-morning','we-anytime', 'wd-anytime', 'we-evening', 'we-morning', 'wd-afternoon']

keras.utils.text_dataset_from_directory(
    dataset,
    labels="inferred",
    label_mode="int",
    class_names=locations,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
)