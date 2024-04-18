"""
Business logic for training component.
"""

from abc import ABC, abstractmethod
import os
import shutil
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from time import time
from typing import Any

import cv2
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras.src.engine.input_layer import Input
from keras.src.layers.activation.leaky_relu import LeakyReLU
from keras.src.engine.training import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.utils.image_utils import img_to_array, load_img
import keras
import tensorflow as tf
from keras import regularizers
from keras.models import load_model
from keras import utils

from shared.common import (FramePreprocessorOpenCV,
                           FramePreprocessorPIL,
                           get_image_shape)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class PreprocessingStage():
    """
    This class defines the preprocessing stage.
    """

    def __init__(self,
                 training_start_folder: str,
                 training_folder: str,
                 preprocessing_setup: dict[str, Any] | None,
                 processing_library: str) -> None:
        self.training_start_folder = training_start_folder
        self.training_folder = training_folder
        self.preprocessing_setup = preprocessing_setup
        self.processing_library = processing_library

    def process(self):
        """
        Apply preprocessing to the training
        images inside training_start_folder.
        The preprocessing setup is defined in preprocessing_setup dictionary.
        """
        try:
            if os.path.exists(self.training_folder):
                # Remove the already existing folder
                shutil.rmtree(self.training_folder)
            # Create production folder
            os.mkdir(self.training_folder)
            for subf in os.listdir(self.training_start_folder):
                src = os.path.join(self.training_start_folder, subf)
                dst = os.path.join(self.training_folder, subf)
                # Create the empty caliper subfolder if does not exist
                if not os.path.isdir(dst):
                    shutil.copytree(src, dst)
            print("Production folder "
                  f"{self.training_folder} created!")

        except OSError as e:
            print(f"Error: {self.training_folder} not copied. {e}")

        if self.processing_library is None:
            print(
                'No preprocessing library has been specified, '
                'skipping preprocessing...')
        else:

            # Apply preprocessing to all images in the training folder
            if self.processing_library == "OpenCV":

                print('Using OpenCV for preprocessing images in'
                      f' {self.training_folder}')
                for filename in os.listdir(self.training_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png") \
                        or filename.endswith(".tiff") \
                            or filename.endswith(".bmp"):
                        image_path = os.path.join(self.training_folder,
                                                  filename)
                        image = cv2.imread(image_path)
                        # Apply preprocessing 1
                        image = FramePreprocessorOpenCV().preprocess_frame(
                                frame=image,
                                preprocessing_setup=self.preprocessing_setup)
                        cv2.imwrite(image_path, image)
                        # Apply augmentation
                        print(f"Performing augmentation on {filename}...")
                        (rotated_frames_list, translated_frames_list,
                            rototranslated_frames_list, luminance_list) = \
                            FramePreprocessorOpenCV().augment_data(
                                image, self.preprocessing_setup)
                        index = 1
                        for augmented in rotated_frames_list:
                            filename_path = os.path.join(
                                self.training_folder,
                                filename[:-4] + f"_rot{index}.bmp")
                            cv2.imwrite(filename_path, augmented)
                            index += 1
                        for augmented in translated_frames_list:
                            filename_path = os.path.join(
                                self.training_folder,
                                filename[:-4] + f"_trasl{index}.bmp")
                            cv2.imwrite(filename_path, augmented)
                            index += 1
                        for augmented in rototranslated_frames_list:
                            filename_path = os.path.join(
                                self.training_folder,
                                filename[:-4] + f"_rotrasl{index}.bmp")
                            cv2.imwrite(filename_path, augmented)
                            index += 1
                        for augmented in luminance_list:
                            filename_path = os.path.join(
                                self.training_folder,
                                filename[:-4] + f"_lum{index}.bmp")
                            cv2.imwrite(filename_path, augmented)
                            index += 1

            elif self.processing_library == "PIL":
                print(
                    'Using Pillow for preprocessing images in'
                    f'{self.training_folder}')
                for filename in os.listdir(self.training_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png") \
                        or filename.endswith(".tiff") \
                            or filename.endswith(".bmp"):
                        image_path = os.path.join(
                            self.training_folder, filename)
                        image = cv2.imread(image_path)
                        # Apply preprocessing
                        image = FramePreprocessorPIL().preprocess_frame(
                            frame=image,
                            preprocessing_setup=self.preprocessing_setup)
                        cv2.imwrite(image_path, image)

            print('Preprocessing has been applied to images in'
                  f' {self.training_folder}')


class BuildStage(ABC):
    """
    Abstract class providing a common interface for build stages.
    """
    @abstractmethod
    def set_initial_network(self):
        """
        Set the initial network for the build stage object.
        """
    @abstractmethod
    def get_initial_network(self):
        """
        Get the initial network from the build stage object.
        """
    @abstractmethod
    def process(self, initial_network):
        """
        Run the build stage.
        """


class TrainStage(ABC):
    """
    Abstract class providing a common interface for train stages.
    """
    @abstractmethod
    def process(self, model):
        """
        Run the train stage.
        """


class AutoencoderBuildStage(BuildStage):
    """
    This class implements the build stage for the autoencoder model.
    """

    def __init__(self,
                 build_setup: dict[str, Any] | None,
                 training_setup: dict[str, Any] | None,
                 weights_folder,
                 preprocessing_setup: dict[str, Any]) -> None:
        self.build_setup = build_setup
        self.training_setup = training_setup
        self.preprocessing_setup = preprocessing_setup
        self.weights_folder = weights_folder
        self.autoencoder = None
        self.initial_network = None

    def set_initial_network(self):
        """
        Define the autoencoder architecture following ispiration
        by what described into "Improving Unsupervised Defect Segmentation
        by Applying Structural Similarity in Autoencoders" paper
        by P.Bergmann et al.
        """
        print("Defining initial network for autoencoder build stage...")

        num_channels = get_image_shape(
            preprocessing_setup=self.preprocessing_setup)[1]
        input_shape = (
            get_image_shape(
                preprocessing_setup=self.preprocessing_setup)[0][0],
            get_image_shape(
                preprocessing_setup=self.preprocessing_setup)[0][1],
            num_channels)

        base_number_of_filters = 32
        latent_space_dim = 10  # 50
        alpha_leakyrelu = 0.25

        input_img = Input(shape=input_shape)
        # Encoder
        h = Conv2D(base_number_of_filters, (3, 3), strides=2,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(input_img)
        h = Conv2D(base_number_of_filters, (3, 3), strides=2,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters, (3, 3), strides=1,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters*2, (3, 3), strides=2,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters*2, (3, 3), strides=1,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters*4, (3, 3), strides=2,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters*2, (3, 3), strides=1,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2D(base_number_of_filters, (3, 3), strides=1,
                   activation=LeakyReLU(alpha=alpha_leakyrelu),
                   padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(h)
        encoded = Conv2D(latent_space_dim, (3, 3), strides=1,
                         activation='linear', padding='valid',
                         kernel_regularizer=regularizers.l2(1e-5))(h)
        # Decoder
        h = Conv2DTranspose(base_number_of_filters, (3, 3), strides=1,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='valid',
                            kernel_regularizer=regularizers.l2(1e-5))(encoded)
        h = Conv2DTranspose(base_number_of_filters*2, (3, 3), strides=1,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters*4, (3, 3), strides=1,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters*2, (3, 3), strides=2,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters*2, (3, 3), strides=1,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters, (3, 3), strides=2,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters, (3, 3), strides=1,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        h = Conv2DTranspose(base_number_of_filters, (3, 3), strides=2,
                            activation=LeakyReLU(alpha=alpha_leakyrelu),
                            padding='same',
                            kernel_regularizer=regularizers.l2(1e-5))(h)
        decoded = Conv2DTranspose(num_channels, (3, 3), strides=2,
                                  activation='linear', padding='same',
                                  kernel_regularizer=regularizers.l2(1e-5))(h)

        model_architecture = Model(input_img, decoded)
        print("Initial network (autoencoder) has been defined!")

        # Save the architecture summary to a log file
        log_filepath = os.path.join(
            self.weights_folder, 'autoencoder_architecture_log.txt')
        # Verify if the file exists
        if not os.path.exists(os.path.dirname(log_filepath)):
            os.makedirs(os.path.dirname(log_filepath))
        with open(log_filepath, 'w') as log_file:
            model_architecture.summary(
                print_fn=lambda x: log_file.write(x + '\n'))

        self.initial_network = model_architecture

    def get_initial_network(self):
        """
        Get the initial network from build stage.
        """
        return self.initial_network

    def process(self, initial_network):
        """
        Build the autoencoder model.
        """
        self.autoencoder = initial_network

        # Choose the optimizer for the autoencoder
        if self.build_setup["optimizer"] == "Adam":
            opt = Adam(learning_rate=self.training_setup["learning_rate"])
        elif self.build_setup["optimizer"] == "SGD":
            opt = SGD(learning_rate=self.training_setup["learning_rate"])
        elif self.build_setup["optimizer"] == "RMSprop":
            opt = RMSprop(learning_rate=self.training_setup["learning_rate"])

        # Choose the loss function for the autoencoder
        if self.build_setup["loss"] == "ssim":
            def ssim_loss(gt, y_pred, max_val=1.0):
                """
                Define the SSIM loss function.
                """
                return (1 - tf.reduce_mean(tf.image.ssim(
                    gt, y_pred, max_val=max_val)))
            # Compile the model
            self.autoencoder.compile(optimizer=opt, loss=ssim_loss)

        elif self.build_setup["loss"] == "mean_squared_error":
            # Example: Add support for Mean Squared Error loss
            def mse_loss(gt, y_pred):
                """
                Define the Mean Squared Error loss function.
                """
                return keras.losses.mean_squared_error(gt, y_pred)
            # Compile the model
            self.autoencoder.compile(optimizer=opt, loss=mse_loss)

        return self.autoencoder


class AutoencoderTrainStage(TrainStage):
    """
    This class implements the train stage for the autoencoder model.
    """

    def __init__(self,
                 preprocessing_setup: dict[str, Any],
                 training_setup: dict[str, Any] | None,
                 training_folder: str,
                 weights_folder: str,
                 ) -> None:
        self.preprocessing_setup = preprocessing_setup
        self.training_setup = training_setup
        self.training_folder = training_folder
        self.weights_folder = weights_folder

    def process(self, model):
        """
        Train the autoencoder model.
        """

        # TODO Disable MKL optimization
        os.environ['TF_DISABLE_MKL'] = '1'

        # Get list of training images
        image_filenames = os.listdir(self.training_folder)
        image_filenames = [filename for filename in image_filenames
                           if not filename.endswith('.txt')]
        # Get number of training images
        num_images = len(image_filenames)

        num_channels = get_image_shape(
            preprocessing_setup=self.preprocessing_setup)[1]
        input_shape = (
            get_image_shape(
                preprocessing_setup=self.preprocessing_setup)[0][0],
            get_image_shape(
                preprocessing_setup=self.preprocessing_setup)[0][1])

        # Initialize the data structure
        x_train = np.empty((num_images,) + input_shape)

        for i, image_filename in enumerate(image_filenames):
            # Get whole image path
            image_path = os.path.join(
                self.training_folder, image_filename)
            if num_channels == 3:
                # Load image to PIL format
                image = load_img(image_path)
                # Convert the image to numpy array and save it inside x_train
                x_train[i] = img_to_array(image)
            elif num_channels == 1:
                # Load the image in grayscale mode using PIL
                image = Image.open(image_path)
                # Convert the image to a numpy array and save it inside x_train
                x_train[i, :, :] = np.array(image.convert("L"))

        # Normalize the image data
        x_train = x_train.astype('float32') / 255.0

        # Split the data into training and validation sets
        x_train, x_val, _, _ = train_test_split(
            x_train, x_train,
            test_size=self.training_setup['validation_split'],
            random_state=42)

        # Reshape the data to fit the model
        x_train = x_train.reshape(
            x_train.shape[0], x_train.shape[1], x_train.shape[2], num_channels)
        x_val = x_val.reshape(
            x_val.shape[0], x_val.shape[1], x_val.shape[2], num_channels)

        exist_checkpoint = False
        checkpoint_filepath = os.path.join(self.weights_folder,
                                           'checkpoint.ckpt')

        def ssim_loss(gt, y_pred, max_val=1.0):
            """
            Define the SSIM loss function.
            """
            return (1 - tf.reduce_mean(tf.image.ssim(
                gt, y_pred, max_val=max_val)))

        if not exist_checkpoint:
            # Save checkpoints
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               mode='min',
                                               verbose=1)
            # Create the weights folder if it does not exist
            if not os.path.exists(self.weights_folder):
                os.makedirs(self.weights_folder)

            train_time_path = os.path.join(self.weights_folder,
                                           'train_time.txt')
            start = time()
            # Train the model
            history = model.fit(x=x_train,
                                y=x_train,
                                epochs=self.training_setup["epochs"],
                                batch_size=self.training_setup["batch_size"],
                                shuffle=True,
                                validation_data=(x_val, x_val),
                                callbacks=[EarlyStopping(
                                    monitor='val_loss',
                                    patience=self.training_setup["patience"]),
                                    model_checkpoint])
            # Save the timing
            with open(train_time_path, 'w') as file:
                file.write(str(time()-start) + '\n')

            # Save the weights
            model = load_model(checkpoint_filepath,
                               custom_objects={'ssim_loss': ssim_loss})
            model.save_weights(os.path.join(self.weights_folder, 'weights.h5'))

        else:
            opt = Adam(learning_rate=self.training_setup["learning_rate"])

            utils.get_custom_objects()['ssim_loss'] = ssim_loss

            # Load the model saved in checkpoint folder
            model = load_model(checkpoint_filepath,
                               custom_objects={'ssim_loss': ssim_loss})

            # Compile the model
            model.compile(optimizer=opt, loss=ssim_loss)

            # Train the model
            history = model.fit(x=x_train,
                                y=x_train,
                                epochs=300,  # Remaining epochs
                                batch_size=4,
                                shuffle=True,
                                validation_data=(x_val, x_val),
                                callbacks=[EarlyStopping(
                                    monitor='val_loss',
                                    patience=10)])

        train_loss_path = os.path.join(self.weights_folder,
                                       'train_loss.txt')
        val_loss_path = os.path.join(self.weights_folder,
                                     'val_loss.txt')

        with open(train_loss_path, 'w') as file:
            for value in history.history['loss']:
                file.write(str(value) + '\n')

        with open(val_loss_path, 'w') as file:
            for value in history.history['val_loss']:
                file.write(str(value) + '\n')


class TrainingStage():
    """
    This class defines the training stage.
    """

    def __init__(
            self,
            build_stage: BuildStage,
            train_stage: TrainStage = None) -> None:

        self.build_stage = build_stage
        self.train_stage = train_stage
        self.model = None
        self.initial_network = None

    def set_initial_network(self):
        """
        Get initial network from build stage.
        """
        # Set the initial network for the build stage object
        self.build_stage.set_initial_network()
        # Get the initial network from the build stage object
        self.initial_network = self.build_stage.get_initial_network()

    def process(self):
        """
        Training stage building and execution.
        """
        # Set the initial network for the training stage object
        self.set_initial_network()

        # Build and train the model.
        # It is a pipeline itself, even though the steps are correlated

        # Run the build stage
        self.model = self.build_stage.process(
            initial_network=self.initial_network
        )

        # Run the train stage
        self.train_stage.process(self.model)
