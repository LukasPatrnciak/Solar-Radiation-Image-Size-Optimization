#  S K Y  I M A G E
#       S I Z E  O P T I M I Z A T I O N
# -------------------------------
# Authors: Bc. Lukas Patrnciak
#          Bc. Andrej Tomcik
#          Bc. Juraj Sevcik



#
# L I B R A R I E S
#
import os
import math
import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import product
from keras import Input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array


#
# I N I T I A L I Z A T I O N
#
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

RAND_ST = 42
random.seed(RAND_ST)
np.random.seed(RAND_ST)
tf.random.set_seed(RAND_ST)


#
# G L O B A L  S E T T I N G S
#
DATASET_NAME = "dataset"
CSV_FILE_NAME = "meteo_data.csv"

IMAGE_COLUMN = "PictureName"
TARGET_COLUMN = "Irradiance"

NUM_SAMPLES = 5
MAX_EPOCHS = 50

OUTPUT_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


#
# B A S E  F U N C T I O N S
#
def print_separator(char="=", length=100):
    print(char * length)


def safe_rmse(mse_value):
    return float(math.sqrt(float(mse_value)))

def analyze_data(dataset_root, target_size):
    sets = os.listdir(dataset_root)

    for st in sets:
        set_path = os.path.join(dataset_root, st)

        if not os.path.isdir(set_path):
            continue

        print("\n====================================")
        print(f"SET: {st}")
        print("====================================")

        images_path = os.path.join(set_path, "images")
        csv_path = os.path.join(set_path, CSV_FILE_NAME)

        if not os.path.exists(images_path):
            print(f"Missing images folder: {images_path}")
            continue

        image_files = []

        for file in os.listdir(images_path):
            lower_name = file.lower()

            if lower_name.endswith(".png") or lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
                image_files.append(file)

        print(f"Images count: {len(image_files)}")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            print(f"CSV shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            if TARGET_COLUMN in df.columns:
                print(f"\n{TARGET_COLUMN} statistics:")
                print(df[TARGET_COLUMN].describe())

        else:
            print(f"Missing CSV file: {csv_path}")

        if len(image_files) > 0:
            selected_images = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

            plt.figure(figsize=(15, 4))

            for idx, image_name in enumerate(selected_images):
                image_path = os.path.join(images_path, image_name)
                img = load_img(image_path, target_size=target_size)

                plt.subplot(1, len(selected_images), idx + 1)
                plt.imshow(img)
                plt.title(image_name, fontsize=8)
                plt.axis("off")

            plt.suptitle(f"Sample images - {st}")
            plt.tight_layout()
            plt.show()


def load_split_dataframe(dataset_root, split_name):
    split_path = os.path.join(dataset_root, split_name)
    csv_path = os.path.join(split_path, CSV_FILE_NAME)
    images_path = os.path.join(split_path, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found: " + csv_path)

    if not os.path.exists(images_path):
        raise FileNotFoundError("Images folder not found: " + images_path)

    df = pd.read_csv(csv_path)

    if IMAGE_COLUMN not in df.columns:
        raise ValueError("Missing column: " + IMAGE_COLUMN)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Missing column: " + TARGET_COLUMN)

    image_paths = []

    for image_name in df[IMAGE_COLUMN]:
        full_path = os.path.join(images_path, str(image_name))
        image_paths.append(full_path)

    df["image_path"] = image_paths
    df = df.dropna(subset=[TARGET_COLUMN])
    valid_rows = []

    for i in range(len(df)):
        path = df.iloc[i]["image_path"]

        if os.path.exists(path):
            valid_rows.append(True)

        else:
            valid_rows.append(False)

    missing_count = valid_rows.count(False)

    if missing_count > 0:
        print("WARNING:", missing_count, "missing images. They will be removed.")
        df = df[valid_rows]

    df = df.reset_index(drop=True)

    print("OK: Loaded split", split_name, "with", len(df), "samples.")

    return df


def fit_target_scaler(train_targets):
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets))

    if target_std == 0.0:
        target_std = 1.0

    return target_mean, target_std


def normalize_targets(values, target_mean, target_std):
    return (values - target_mean) / target_std


def denormalize_targets(values, target_mean, target_std):
    return values * target_std + target_mean


def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0

    return img


def create_tensorflow_dataset(dataframe, image_size, batch_size, shuffle=True, repeat=False):
    image_paths = dataframe["image_path"].values
    target_values = dataframe[TARGET_COLUMN].values.astype(np.float32)

    def generator():
        for i in range(len(image_paths)):
            path = image_paths[i]
            target = target_values[i]
            image = load_and_preprocess_image(path, image_size)

            yield image, target

    height = image_size[0]
    width = image_size[1]
    channels = 3

    image_shape = (height, width, channels)

    image_spec = tf.TensorSpec(shape=image_shape, dtype=tf.float32)
    target_spec = tf.TensorSpec(shape=(), dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(image_spec, target_spec))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256, seed=RAND_ST)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_small_cnn(learning_rate, dropout_rate, dense_units, input_shape):
    model = models.Sequential([
        Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model


def create_medium_cnn(learning_rate, dropout_rate, dense_units, input_shape):
    model = models.Sequential([
        Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model


def create_large_cnn(learning_rate, dropout_rate, dense_units, input_shape):
    model = models.Sequential([
        Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model


def create_cnn_model(architecture_name, learning_rate, dropout_rate, dense_units, input_shape):
    if architecture_name == "small":
        return create_small_cnn(learning_rate, dropout_rate, dense_units, input_shape)

    if architecture_name == "medium":
        return create_medium_cnn(learning_rate, dropout_rate, dense_units, input_shape)

    if architecture_name == "large":
        return create_large_cnn(learning_rate, dropout_rate, dense_units, input_shape)

    raise ValueError(f"Unknown architecture name: {architecture_name}")


def train_cnn_model(architecture_name, learning_rate, dropout_rate, dense_units, image_size, batch_size, epochs):
    input_shape = (image_size[0], image_size[1], 3)

    model = create_cnn_model(
        architecture_name=architecture_name,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        input_shape=input_shape
    )

    drop_str = str(dropout_rate).replace(".", "_")
    lr_str = str(learning_rate).replace(".", "_")

    model_name = (
        f"{architecture_name}_{image_size[0]}x{image_size[1]}"
        f"_dense{dense_units}"
        f"_drop{drop_str}"
        f"_bs{batch_size}"
        f"_lr{lr_str}"
    )

    checkpoint_path = os.path.join(MODELS_DIR, model_name + ".keras")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )

    train_dataset = create_tensorflow_dataset(
        train_df,
        image_size,
        batch_size,
        shuffle=True,
        repeat=True
    )

    val_dataset = create_tensorflow_dataset(
        val_df,
        image_size,
        batch_size,
        shuffle=False,
        repeat=False
    )

    test_dataset = create_tensorflow_dataset(
        test_df,
        image_size,
        batch_size,
        shuffle=False,
        repeat=False
    )

    train_steps = math.ceil(len(train_df) / batch_size)
    val_steps = math.ceil(len(val_df) / batch_size)

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    end_time = time.time()
    training_time_seconds = end_time - start_time
    epochs_trained = len(history.history["loss"])

    train_loss, train_mae = model.evaluate(
        train_dataset,
        steps=train_steps,
        verbose=0
    )

    val_loss, val_mae = model.evaluate(
        val_dataset,
        verbose=0
    )

    test_loss, test_mae = model.evaluate(
        test_dataset,
        verbose=0
    )

    return model, history, train_mae, val_mae, test_mae, train_loss, val_loss, test_loss, training_time_seconds, epochs_trained


def evaluate_model_original_scale(model, dataframe, image_size, batch_size, target_mean, target_std):
    dataset = create_tensorflow_dataset(dataframe, image_size, batch_size, shuffle=False)

    y_true_norm = []
    y_pred_norm = []

    for image_batch, target_batch in dataset:
        predictions = model.predict(image_batch, verbose=0).reshape(-1)

        y_true_norm.extend(target_batch.numpy().reshape(-1))
        y_pred_norm.extend(predictions)

    y_true_norm = np.array(y_true_norm, dtype=np.float32)
    y_pred_norm = np.array(y_pred_norm, dtype=np.float32)

    y_true = denormalize_targets(y_true_norm, target_mean, target_std)
    y_pred = denormalize_targets(y_pred_norm, target_mean, target_std)

    mse_original = float(np.mean((y_true - y_pred) ** 2))
    mae_original = float(np.mean(np.abs(y_true - y_pred)))
    rmse_original = safe_rmse(mse_original)

    return mse_original, mae_original, rmse_original, y_true, y_pred


def plot_training(history, title_prefix=""):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title(f"{title_prefix} Training and Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{title_prefix} Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_predictions_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Irradiance")
    plt.ylabel("Predicted Irradiance")
    plt.title(title)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.show()


def plot_metric_vs_image_size(results_df, metric_name, title):
    plt.figure(figsize=(10, 6))

    architectures = sorted(results_df["architecture"].unique())

    for architecture_name in architectures:
        architecture_data = results_df[results_df["architecture"] == architecture_name]
        architecture_data = architecture_data.sort_values("image_height")

        x_values = architecture_data["image_height"]
        y_values = architecture_data[metric_name]

        plt.plot(x_values, y_values, marker="o", label=architecture_name)

    plt.xlabel("Image size")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_time_vs_image_size(results_df, title):
    plt.figure(figsize=(10, 6))

    architectures = results_df["architecture"].unique()
    architectures = sorted(architectures)

    for architecture_name in architectures:
        architecture_data = results_df[results_df["architecture"] == architecture_name]

        architecture_data = architecture_data.sort_values("image_height")
        architecture_data = architecture_data.drop_duplicates(subset="image_height", keep="first")

        x_values = architecture_data["image_height"]
        y_values = architecture_data["training_time_seconds"]

        plt.plot(x_values, y_values, marker="o", label=architecture_name)

    plt.xlabel("Image size")
    plt.ylabel("Training time (seconds)")
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()


#
# R U N
#
analyze_data(DATASET_NAME, (128, 128))

train_df = load_split_dataframe(DATASET_NAME, "train")
val_df = load_split_dataframe(DATASET_NAME, "val")
test_df = load_split_dataframe(DATASET_NAME, "test")

fit_target_mean, fit_target_std = fit_target_scaler(train_df[TARGET_COLUMN].values)

print(f"\nTarget mean: {fit_target_mean}")
print(f"Target std: {fit_target_std}")

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df[TARGET_COLUMN] = normalize_targets(train_df[TARGET_COLUMN].values, fit_target_mean, fit_target_std)
val_df[TARGET_COLUMN] = normalize_targets(val_df[TARGET_COLUMN].values, fit_target_mean, fit_target_std)
test_df[TARGET_COLUMN] = normalize_targets(test_df[TARGET_COLUMN].values, fit_target_mean, fit_target_std)

sample_dataset = create_tensorflow_dataset(train_df, (128, 128), 16, shuffle=True)

for images, targets in sample_dataset.take(1):
    print("\nDataset check:")
    print("Images shape:", images.shape)
    print("Targets shape:", targets.shape)
    print("Images dtype:", images.dtype)
    print("Targets dtype:", targets.dtype)

# EXPERIMENT SETTINGS
architectures_list = ["small", "medium", "large"]
image_sizes_list = [(64, 64), (96, 96), (128, 128), (160, 160), (224, 224)]
learning_rates_list = [0.001, 0.0005]
dropout_rates_list = [0.2, 0.3]
dense_units_list = [64, 128]
batch_sizes_list = [16, 32]

all_experiments = list(product(
    architectures_list,
    image_sizes_list,
    learning_rates_list,
    dropout_rates_list,
    dense_units_list,
    batch_sizes_list
))

print_separator()
print(f"Total number of experiments: {len(all_experiments)}")
print_separator()

results = []

for experiment_index, experiment in enumerate(all_experiments, start=1):
    architecture_name_element = experiment[0]
    image_size_element = experiment[1]
    learning_rate_element = experiment[2]
    dropout_rate_element = experiment[3]
    dense_units_element = experiment[4]
    batch_size_element = experiment[5]

    print("\n")
    print_separator()
    print(f"EXPERIMENT {experiment_index} / {len(all_experiments)}")
    print(f"architecture={architecture_name_element}")
    print(f"image_size={image_size_element}")
    print(f"learning_rate={learning_rate_element}")
    print(f"dropout_rate={dropout_rate_element}")
    print(f"dense_units={dense_units_element}")
    print(f"batch_size={batch_size_element}")
    print_separator()

    trained_experiment_model = train_cnn_model(
        architecture_name=architecture_name_element,
        learning_rate=learning_rate_element,
        dropout_rate=dropout_rate_element,
        dense_units=dense_units_element,
        image_size=image_size_element,
        batch_size=batch_size_element,
        epochs=MAX_EPOCHS
    )

    experiment_model = trained_experiment_model[0]
    experiment_history = trained_experiment_model[1]
    experiment_train_mae = trained_experiment_model[2]
    experiment_val_mae = trained_experiment_model[3]
    experiment_test_mae = trained_experiment_model[4]
    experiment_train_loss = trained_experiment_model[5]
    experiment_val_loss = trained_experiment_model[6]
    experiment_test_loss = trained_experiment_model[7]
    experiment_training_time = trained_experiment_model[8]
    experiment_epochs_trained = trained_experiment_model[9]

    val_mse_original, val_mae_original, val_rmse_original, _, _ = evaluate_model_original_scale(
        experiment_model, val_df, image_size_element, batch_size_element, fit_target_mean, fit_target_std
    )

    test_mse_original, test_mae_original, test_rmse_original, test_y_true, test_y_pred = evaluate_model_original_scale(
        experiment_model, test_df, image_size_element, batch_size_element, fit_target_mean, fit_target_std
    )

    results.append({
        "architecture": architecture_name_element,
        "model": experiment_model,
        "history": experiment_history,
        "image_height": image_size_element[0],
        "image_width": image_size_element[1],
        "image_size_label": f"{image_size_element[0]}x{image_size_element[1]}",
        "learning_rate": learning_rate_element,
        "dropout_rate": dropout_rate_element,
        "dense_units": dense_units_element,
        "batch_size": batch_size_element,
        "epochs_trained": experiment_epochs_trained,
        "training_time_seconds": experiment_training_time,

        "train_mae_norm": experiment_train_mae,
        "val_mae_norm": experiment_val_mae,
        "test_mae_norm": experiment_test_mae,
        "train_loss_norm": experiment_train_loss,
        "val_loss_norm": experiment_val_loss,
        "test_loss_norm": experiment_test_loss,

        "val_mse_original": val_mse_original,
        "val_mae_original": val_mae_original,
        "val_rmse_original": val_rmse_original,

        "test_mse_original": test_mse_original,
        "test_mae_original": test_mae_original,
        "test_rmse_original": test_rmse_original,

        "test_y_true": test_y_true,
        "test_y_pred": test_y_pred
    })

    print(f"Train MAE (normalized): {experiment_train_mae:.4f}")
    print(f"Val   MAE (normalized): {experiment_val_mae:.4f}")
    print(f"Test  MAE (normalized): {experiment_test_mae:.4f}")

    print(f"Val   MAE (original): {val_mae_original:.4f}")
    print(f"Val   RMSE(original): {val_rmse_original:.4f}")
    print(f"Test  MAE (original): {test_mae_original:.4f}")
    print(f"Test  RMSE(original): {test_rmse_original:.4f}")

    print(f"Training time (s): {experiment_training_time:.2f}")
    print(f"Epochs trained:    {experiment_epochs_trained}")


#
# R E S U L T S
#
results_dataframe = pd.DataFrame(results)
results_printable = results_dataframe.drop(columns=["model", "history", "test_y_true", "test_y_pred"]).sort_values(by="val_mae_original", ascending=True).reset_index(drop=True)

print_separator()
print("RESULTS TABLE")
print_separator()
print(results_printable)

results_save_path = os.path.join(TABLES_DIR, "results.csv")
results_printable.to_csv(results_save_path, index=False)

print(f"\nSaved results table: {results_save_path}")


#
# B E S T  M O D E L
#
best_index = results_dataframe["val_mae_original"].idxmin()
best_model_row = results_dataframe.loc[best_index]

best_model = best_model_row["model"]
best_history = best_model_row["history"]

best_architecture = best_model_row["architecture"]
best_image_size_label = best_model_row["image_size_label"]
best_learning_rate = best_model_row["learning_rate"]
best_dropout_rate = best_model_row["dropout_rate"]
best_dense_units = best_model_row["dense_units"]
best_batch_size = best_model_row["batch_size"]
best_epochs_trained = best_model_row["epochs_trained"]
best_training_time = best_model_row["training_time_seconds"]

best_val_mae_original = best_model_row["val_mae_original"]
best_val_rmse_original = best_model_row["val_rmse_original"]
best_test_mae_original = best_model_row["test_mae_original"]
best_test_rmse_original = best_model_row["test_rmse_original"]

best_test_y_true = best_model_row["test_y_true"]
best_test_y_pred = best_model_row["test_y_pred"]

print_separator()
print("BEST MODEL (based on Validation MAE in original scale)")
print_separator()
print(f"architecture       = {best_architecture}")
print(f"image_size         = {best_image_size_label}")
print(f"learning_rate      = {best_learning_rate}")
print(f"dropout_rate       = {best_dropout_rate}")
print(f"dense_units        = {best_dense_units}")
print(f"batch_size         = {best_batch_size}")
print(f"epochs_trained     = {best_epochs_trained}")
print(f"training_time_sec  = {best_training_time:.2f}")
print(f"val_mae_original   = {best_val_mae_original:.4f}")
print(f"val_rmse_original  = {best_val_rmse_original:.4f}")
print(f"test_mae_original  = {best_test_mae_original:.4f}")
print(f"test_rmse_original = {best_test_rmse_original:.4f}")
print_separator()

best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
best_model.save(best_model_path)

best_model_summary = best_model_row.drop(labels=["model", "history", "test_y_true", "test_y_pred"])
best_model_summary = pd.DataFrame([best_model_summary])
best_model_summary_path = os.path.join(TABLES_DIR, "best_model_summary.csv")
best_model_summary.to_csv(best_model_summary_path, index=False)

print(f"Saved best model: {best_model_path}")
print(f"Saved best model summary: {best_model_summary_path}")


#
# P L O T S
#
plot_training(best_history, title_prefix=f"{best_architecture} ")
plot_predictions_scatter(best_test_y_true, best_test_y_pred, "Best Model - Test Predictions vs True Irradiance")
plot_metric_vs_image_size(results_printable, "val_mae_original", "Validation MAE vs Image Size")
plot_metric_vs_image_size(results_printable, "val_rmse_original", "Validation RMSE vs Image Size")
plot_training_time_vs_image_size(results_printable, "Training Time vs Image Size")