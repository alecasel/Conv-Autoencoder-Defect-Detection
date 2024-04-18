"""
Application layer for acquisition component.
"""

import tkinter
from business.acquisition import (
    AcquisitionManager, CameraOpenCV, CameraOpenCVHTTP)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class AcquisitionService():
    """
    Acquisition service class.

    It exposes the basic methods for the acquisition component.
    It also provides a camera attribute
    that can be used to access the camera frames
    """

    def __init__(self,
                 camera_type: str) -> None:
        self.camera_type = camera_type
        self.camera = None

    def manage_acquisition(self,
                           training_folder: str,
                           production_temp_folder: str) -> str:
        """
        Manages the acquisition process by interacting with the user
        to obtain input and create production temporary folders.

        Args:
        - training_folder (str): Path of the training folder.
        - production_folder (str): Path of the production folder.
        - production_temp_folder (str):
        Path of the production temporary folder.

        Returns:
        - caliper_folders_paths (list[str]): List of paths to caliper folders,
          including the production temporary folder.
        """
        root = tkinter.Tk()
        root.withdraw()

        # Get user input using AcquisitionManager
        self.acquisition = AcquisitionManager(training_folder,
                                              production_temp_folder)
        caliper_folder_path = self.acquisition.get_user_input()

        if 'production' in caliper_folder_path:
            # Create the production temp folder associated with the caliper
            self.acquisition.create_temp_folder(caliper_folder_path)

        return caliper_folder_path

    def open_camera_connection(self,
                               camera_id: int,
                               camera_serial_number: str,
                               camera_width: int,
                               camera_height: int,
                               enable_directshow: bool,
                               reversed_position: bool,
                               bounding_box: tuple[int]):
        """
        This method opens a connection to the camera,
        based on the camera type.

        The camera connection is opened
        as a separate thread.
        """
        if self.camera_type == "OpenCV":
            if isinstance(camera_serial_number, str):
                camera_serial_number = int(camera_serial_number)
            self.camera = CameraOpenCV(
                camera_id,
                camera_serial_number,
                camera_width,
                camera_height,
                enable_directshow=enable_directshow)
            if bounding_box["display"]:
                self.camera.display_caliper_box(bounding_box["coords"],
                                                camera_width,
                                                camera_height,
                                                reversed_position)
            self.camera.start_acquisition_thread(reversed_position)
        elif self.camera_type == "OpenCVHTTP":
            if isinstance(camera_serial_number, str):
                camera_serial_number = int(camera_serial_number)
            self.camera = CameraOpenCVHTTP(
                camera_id,
                camera_serial_number,
                camera_width,
                camera_height,
                enable_directshow=enable_directshow)
            self.camera.start_acquisition_thread(reversed_position)
        else:
            raise ValueError("Camera type not supported.")

    def save_image(self,
                   dest_folder: str):
        """
        This method saves an image from the camera stream
        in the training/production folder. It requires the camera
        to be connected via the open_camera_connection method.
        """
        if self.camera is not None:
            self.camera.save_frame(dest_folder)
        else:
            print("Camera not connected!")

    def close_camera_connection(self):
        """
        This method closes the camera connection.
        """
        if self.camera is not None:
            self.camera.stop_acquisition_thread()
        else:
            print("Camera not connected!")


# Test acquisition application logic.
if __name__ == "__main__":

    # Get the parent folder of current file
    current_folder = os.path.dirname(__file__)
    # Get the parent folder of current folder
    app_folder = os.path.dirname(current_folder)

    training_folder_cam1 = os.path.join(
        app_folder, "data", "training_photos", "cam_1")

    # training_folder_cam2 = os.path.join(
    #    app_folder, "data", "training_photos", "cam_2")

    acquisition_cam_1 = AcquisitionService(
        camera_type="OpenCV")

    acquisition_cam_1.open_camera_connection(
        camera_id=0,
        camera_serial_number="0",
        camera_width=640,
        camera_height=480)

    import time

    time.sleep(2.5)
    acquisition_cam_1.save_image(dest_folder=training_folder_cam1)
    time.sleep(2.5)
    acquisition_cam_1.save_image(dest_folder=training_folder_cam1)
