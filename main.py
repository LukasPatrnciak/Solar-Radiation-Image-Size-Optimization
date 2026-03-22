import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input
from keras.src.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")


IMG_HEIGHT=128
IMG_WIDTH=128
FULL_RES_HEIGHT=128
FULL_RES_WIDTH=128
BATCH_SIZE=32
NUM_CLASS_SAMPLES=5
NUM_COMP_PCA=50
NUM_COMP_TSNE=2
RAND_ST=42
NUM_CLUST=10
IMAGES_PER_CLUSTER=5
CNN_EPOCHS=10
NN_EPOCHS=100
TL_EPOCHS=10
FEATURES_DATASET_NAME="features_dataset.csv"


def analyze_data(dataset, max_classes, target_size):
    sets = os.listdir(dataset)
    sets_classes_count = {}

    for st in sets:
        set_path = os.path.join(dataset, st)

        if not os.path.isdir(set_path):
            continue

        classes = os.listdir(set_path)
        class_counts = {}

        selected_classes = sorted(classes)[:max_classes]

        plt.figure(figsize=(15, 5))

        for idx, cls in enumerate(selected_classes):
            class_path = os.path.join(set_path, cls)

            if not os.path.isdir(class_path):
                continue

            image_count = len(os.listdir(class_path))
            class_counts[cls] = image_count

            sample_image_path = os.path.join(class_path, os.listdir(class_path)[0])
            img = load_img(sample_image_path, target_size=target_size)
            plt.subplot(1, len(selected_classes), idx + 1)
            plt.imshow(img)
            #plt.title(f"{cls} ({image_count})")
            plt.axis('off')

        plt.suptitle(f"Analysed Set: {st}")
        plt.tight_layout()
        plt.show()

        sets_classes_count = len(classes)

        print(f"Classes count in Set '{st}': {sets_classes_count}")
        print(f"Pictures count in Class (based on {max_classes} classes): {class_counts}")
        print("\n")

    return sets_classes_count



if __name__ == '__main__':
    sun_radiation_dataset = "dataset"
    classes_count = analyze_data(sun_radiation_dataset, NUM_CLASS_SAMPLES, (FULL_RES_HEIGHT, FULL_RES_WIDTH))

    print(f"Count of ouput Neurons in  CNN: {classes_count}")

    train_dir = sun_radiation_dataset + '/train'
    val_dir = sun_radiation_dataset + '/val'
    test_dir = sun_radiation_dataset + '/test'
