import os
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")



NUM_IMAGES = 5
IMAGE_COLUMN = "PictureName"
TARGET_COLUMN = "Irradiance"



def analyze_dataset(dataset_root, target_size):
    sets = os.listdir(dataset_root)

    for st in sets:
        set_path = os.path.join(dataset_root, st)

        if not os.path.isdir(set_path):
            continue

        print("\n==============================")
        print("SET:", st)
        print("==============================")

        # ----------------------------------
        # IMAGES
        # ----------------------------------
        images_path = os.path.join(set_path, "images")

        if not os.path.exists(images_path):
            print("Chýba priečinok images")
            continue

        image_files = []
        all_files = os.listdir(images_path)

        for file_name in all_files:
            file_name_lower = file_name.lower()

            if file_name_lower.endswith(".png") or file_name_lower.endswith(".jpg") or file_name_lower.endswith(".jpeg"):
                image_files.append(file_name)

        print("Počet obrázkov:", len(image_files))

        if len(image_files) > 0:
            sample_files = random.sample(image_files, min(NUM_IMAGES, len(image_files)))

            plt.figure(figsize=(15, 3))

            for idx, file in enumerate(sample_files):
                img_path = os.path.join(images_path, file)
                img = load_img(img_path, target_size=target_size)

                plt.subplot(1, len(sample_files), idx + 1 )
                plt.imshow(img)
                plt.title(file, fontsize=8)
                plt.axis("off")

            plt.suptitle(f"Sample images - {st}")
            plt.tight_layout()
            plt.show()

        # ----------------------------------
        # CSV FILES
        # ----------------------------------
        csv_files = []
        all_files = os.listdir(set_path)

        for file_name in all_files:
            if file_name.endswith(".csv"):
                csv_files.append(file_name)

        for csv_file in csv_files:
            csv_path = os.path.join(set_path, csv_file)

            print("\nCSV:", csv_file)

            try:
                df = pd.read_csv(csv_path)

                print("(Rows, Columns):", df.shape)
                print("Columns:")
                print(df.columns.tolist())

                print("\nFirst row:")
                print(df.iloc[0])

                # Basic Stats
                if "Irradiance" in df.columns:
                    print("\nIrradiance stats:")
                    print(df["Irradiance"].describe())

                print()

            except Exception as e:
                print("Chyba pri načítaní CSV:", e)


def load_split_dataframe(dataset_root, split_name, csv_file_name):
    split_path = os.path.join(dataset_root, split_name)
    csv_path = os.path.join(str(split_path), csv_file_name)
    images_path = os.path.join(str(split_path), "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Chýba CSV súbor: {csv_path}")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Chýba priečinok images: {images_path}")

    df = pd.read_csv(csv_path)

    if IMAGE_COLUMN not in df.columns:
        raise ValueError(f"V CSV chýba stĺpec '{IMAGE_COLUMN}'. "f"Dostupné stĺpce: {list(df.columns)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"V CSV chýba stĺpec '{TARGET_COLUMN}'. "f"Dostupné stĺpce: {list(df.columns)}")

    image_paths = []

    for image_name in df[IMAGE_COLUMN]:
        full_image_path = os.path.join(images_path, str(image_name))
        image_paths.append(full_image_path)

    df["image_path"] = image_paths

    return df


def check_dataset(df, split_name):
    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATASET")
    print("=" * 60)

    print("Shape:", df.shape)
    print("\nStĺpce:")
    print(df.columns.tolist())

    print("\nPrvých 5 riadkov:")
    print(df[[IMAGE_COLUMN, TARGET_COLUMN, "image_path"]].head())

    existing_images = 0
    missing_images = 0

    for path in df["image_path"]:
        if os.path.exists(path):
            existing_images += 1
        else:
            missing_images += 1

    print("\nPočet existujúcich obrázkov:", existing_images)
    print("Počet chýbajúcich obrázkov:", missing_images)

    print("\nŠtatistika targetu Irradiance:")
    print(df[TARGET_COLUMN].describe())


def build_small_cnn(input_shape, learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_medium_cnn(input_shape, learning_rate=0.001, dropout_rate=0.2):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_large_cnn(input_shape, learning_rate=0.001, dropout_rate=0.3):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model


def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0

    return img


def create_tf_dataset(df, image_size, batch, shuffle=True):
    image_paths = df["image_path"].values
    target_values = df["Irradiance"].values

    def generator():
        for path, target in zip(image_paths, target_values):
            img = load_and_preprocess_image(path, image_size)

            yield img, target

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(image_size[0], image_size[1], 3),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.float32
            )
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model(model_name, input_shape, learning_rate=0.001, dropout_rate=0.2):
    if model_name == "small":
        return build_small_cnn(input_shape, learning_rate)

    if model_name == "medium":
        return build_medium_cnn(input_shape, learning_rate, dropout_rate)

    if model_name == "large":
        return build_large_cnn(input_shape, learning_rate, dropout_rate)

    raise ValueError(f"Neznámy model: {model_name}")



if __name__ == "__main__":
    root_dataset = "dataset"
    image_sizes = [(64, 64), (96, 96), (128, 128), (160, 160), (224, 224)]
    initial_shape = (128, 128, 3)
    batch_size = 32

    analyze_dataset(root_dataset, image_sizes[4])

    train_df = load_split_dataframe(root_dataset, "train", "meteo_data_cleaned.csv")
    val_df = load_split_dataframe(root_dataset, "val", "meteo_data_cleaned.csv")
    test_df = load_split_dataframe(root_dataset, "test", "meteo_data_cleaned.csv")

    check_dataset(train_df, "train")
    check_dataset(val_df, "val")
    check_dataset(test_df, "test")

    small_model = build_small_cnn(initial_shape)
    medium_model = build_medium_cnn(initial_shape)
    large_model = build_large_cnn(initial_shape)

    train_ds = create_tf_dataset(
        train_df,
        image_sizes[2],
        batch_size,
        shuffle=True
    )

    val_ds = create_tf_dataset(
        val_df,
        image_sizes[2],
        batch_size,
        shuffle=False
    )

    test_ds = create_tf_dataset(
        test_df,
        image_sizes[2],
        batch_size,
        shuffle=False
    )

    # Dataset Checks
    for images, targets in train_ds.take(1):
        print("Images shape:", images.shape)
        print("Targets shape:", targets.shape)

    # Training
    history = small_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )


