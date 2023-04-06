import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import preprocessing
import pandas as pd
from PIL import Image
from keras.utils import load_img
from keras.utils import img_to_array
# I don't think we need this function
# from keras.utils import array_to_img


training_csv = pd.read_csv('fairface_label_train.csv')
training_csv = training_csv.iloc[1:25]
training_csv = training_csv.reset_index()
training_images = []
for index, row in training_csv.iterrows():
    image = Image.open(row['file']).convert("L") # load_img(row['file'], color_mode="grayscale", target_size=(32,32))
    print(f"Loaded image: {type(image)}")
    print(f"image #{index} file name: {row['file']}")
    print(f"image #{index} format: {image.format}")
    print(f"image #{index} size: {image.size}")
    image_array = np.asarray(image) # img_to_array(image) (using the keras methods gives float32 type instead of uint8, not sure which we need)
    print(f"Numpy array: {type(image_array)}") 
    print(f"image array #{index} type: {image_array.dtype}")
    print(f"image array #{index} shape: {image_array.shape}")


