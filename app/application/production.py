"""
Application logic for production component.
"""

from business.training import TrainingStage, AutoencoderBuildStage
from business.production import (PreprocessingFeatures,
                                 AutoencoderFeatures,
                                 BodyHandFeatures)
from business.acquisition import CameraAcquisition
try:
    from application.acquisition import AcquisitionService
except ImportError:
    from acquisition import AcquisitionService

from pyparsing import Any
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ProductionService():
    """
    This class defines the production service.

    It exposes the production features for the desired application.
    """

    def __init__(self,
                 production_start_folder: str,
                 production_folder: str,
                 weights_folder: str,
                 camera: CameraAcquisition,
                 application: str,
                 initial_network,
                 preprocessing_setup: dict[str, Any] | None = None,
                 processing_library: str = 'OpenCV') -> None:

        self.production_start_folder = production_start_folder
        self.production_folder = production_folder
        self.weights_folder = weights_folder
        self.camera = camera
        self.initial_network = initial_network
        self.preprocessing_setup = preprocessing_setup
        self.processing_library = processing_library
        self.features_1 = None
        self.features_2 = None

        # Instantiate the training stage
        if application == 'Defect Detection':
            self.features_1 = PreprocessingFeatures(
                camera=self.camera,
                production_start_folder=self.production_start_folder,
                production_folder=self.production_folder,
                preprocessing_setup=self.preprocessing_setup,
                processing_library=self.processing_library
            )
            self.features_2 = AutoencoderFeatures(
                production_folder=self.production_folder,
                weights_folder=self.weights_folder,
                initial_network=self.initial_network
            )

        elif application == 'Body/Hand Detection':
            self.features = BodyHandFeatures(
                weights_folder=self.weights_folder,
                camera=self.camera,
            )


# Test production application logic
if __name__ == "__main__":

    # Get the parent folder of current file
    current_folder = os.path.dirname(__file__)
    # Get the parent folder of current folder
    app_folder = os.path.dirname(current_folder)

    weights_folder_cam1 = os.path.join(
        app_folder, "data", "weights", "cam_1")

    production_folder_cam1 = os.path.join(
        app_folder, "data", "production_photos", "cam_1")

    acquisition_cam_1 = AcquisitionService(
        camera_type="OpenCV")
    acquisition_cam_1.open_camera_connection(
        camera_id=1,
        camera_serial_number="0",
        camera_width=640,
        camera_height=480)

    training_stage_cam1 = TrainingStage(
        build_stage=AutoencoderBuildStage(
            build_setup=None,
        ),
        train_stage=None
    )
    training_stage_cam1.set_initial_network()
    initial_network = training_stage_cam1.initial_network

    production_cam_1 = ProductionService(
        weights_folder=weights_folder_cam1,
        camera=acquisition_cam_1.camera,
        application='Defect Detection',
        initial_network=initial_network
    )

    import time

    time.sleep(5)
    print("Saving current frame...")
    production_cam_1.features.save_current_frame(
        production_folder=production_folder_cam1
    )

    print("A request for inference has been received...")
    production_cam_1.features.get_results(
        preprocessing_setup=None,
    )
