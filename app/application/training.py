"""
Application layer for training component.
"""

from shared.common import Pipeline
from business.training import (PreprocessingStage, TrainingStage,
                               AutoencoderBuildStage, AutoencoderTrainStage)
from multiprocessing import Process
import os
import platform
import sys
from threading import Thread
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TrainingService():
    """
    Defines the training service class.

    A training service will be associated to a training folder
    and a pipeline defined for the training process itself.
    The training service contains the method
    to build and start the training pipeline.
    The training pipeline is composed by a series of steps:
    the preprocessing stage, the training stage... etc.
    The preprocessing stage is common to all training pipelines.
    The training stage depends on the application: in fact the build stage
    and the train stage will be different for each application.
    The training pipeline will be executed as a separate process on the host.
    """

    def __init__(self,
                 training_start_folder: str,
                 training_folder: str,
                 weights_folder: str,
                 ) -> None:

        self.training_start_folder = training_start_folder
        self.training_folder = training_folder
        self.weights_folder = weights_folder
        self.pipeline = Pipeline()

    def start_training_pipeline(self,
                                application: str,
                                preprocessing_setup: dict[str,
                                                          Any] | None = None,
                                processing_library: str = 'OpenCV',
                                build_setup: dict[str, Any] | None = None,
                                training_setup: dict[str, Any] | None = None,
                                ):
        """
        Start training pipeline.

        The training pipeline is composed by a series of stages/steps,
        which are defined, instantiated and added to the pipeline.
        The final pipeline is then executed as a separate thread
        or process on the host.
        """

        # Instantiate the preprocessing stage
        preprocessing_stage = PreprocessingStage(
            training_start_folder=self.training_start_folder,
            training_folder=self.training_folder,
            preprocessing_setup=preprocessing_setup,
            processing_library=processing_library
        )

        # Instantiate the training stage
        if application == 'Defect Detection':
            training_stage = TrainingStage(
                build_stage=AutoencoderBuildStage(
                    build_setup=build_setup,
                    training_setup=training_setup,
                    preprocessing_setup=preprocessing_setup,
                    weights_folder=self.weights_folder
                ),
                train_stage=AutoencoderTrainStage(
                    preprocessing_setup=preprocessing_setup,
                    training_setup=training_setup,
                    training_folder=self.training_folder,
                    weights_folder=self.weights_folder,
                )
            )

        # Add the stages to the pipeline
        self.pipeline.add_stage(stage=preprocessing_stage)
        self.pipeline.add_stage(stage=training_stage)

        # Each training pipeline is executed as
        # a separate thread (on Windows) or process on the host
        # (So don't panic if one camera pipeline
        # is excecuted before the other one)
        if platform.system() == 'Linux':
            Process(target=self.pipeline.execute_pipeline).start()
        else:
            Thread(target=self.pipeline.execute_pipeline).start()


# Test training application logic
if __name__ == "__main__":

    # Get the parent folder of current file
    current_folder = os.path.dirname(__file__)
    # Get the parent folder of current folder
    app_folder = os.path.dirname(current_folder)

    training_folder_cam1 = os.path.join(
        app_folder, "data", "training_photos", "cam_1")
    training_folder_cam2 = os.path.join(
        app_folder, "data", "training_photos", "cam_2")

    weights_folder_cam1 = os.path.join(
        app_folder, "data", "weights", "cam_1")
    weights_folder_cam2 = os.path.join(
        app_folder, "data", "weights", "cam_2")

    training_service_cam1 = TrainingService(
        training_folder=training_folder_cam1,
        training_support_folder=training_folder_cam1,
        weights_folder=weights_folder_cam1,
    )

    training_service_cam1.start_training_pipeline(
        application='Defect Detection',
        preprocessing_setup={'resize': (256, 256), 'grayscale': True},
        build_setup={'optimizer': 'adam', 'loss': 'ssim'},
        training_setup={'epochs': 10, 'batch_size': 32, 'learning_rate': 0.001}
    )
