"""
Business logic for camera acquisition.
"""

from abc import ABC, abstractmethod
import glob
import os
import platform
from threading import Event, Thread
from tkinter import simpledialog

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np
import uvicorn


class AcquisitionManager():
    """
    Acquisition Manager class.
    It supports the use of tkinter and
    manages the temporary creation of folders
    where to save the acquired images.
    """

    def __init__(self,
                 training_folder_start_cam1,
                 production_temp_folder_start_cam1):
        self.training_folder_start_cam1 = training_folder_start_cam1
        self.production_temp_folder_start_cam1 = (
            production_temp_folder_start_cam1)

    def get_user_input(self) -> str:
        """
        Gets user input for selecting mode and entering folder names.
        """
        def list_subfolders(folder_chosen_path):
            """
            Lists subfolders in the specified folder.
            Creates the folder if it doesn't exist.
            """
            calipers_list = []
            # Verify the folder's existence
            if (os.path.exists(folder_chosen_path)
                    and os.path.isdir(folder_chosen_path)):
                calipers_list = [folder for folder in os.listdir(
                    folder_chosen_path)
                    if os.path.isdir(os.path.join(folder_chosen_path, folder))]
                return calipers_list
            else:  # If does not exist, then create it
                os.makedirs(folder_chosen_path)
            return calipers_list

        # Ask the user to select the mode
        mode = simpledialog.askstring(
            "Input",
            "Type 'T' for Training or 'P' for Production").lower()

        # Verify the validity of the user input
        while mode not in ['T', 't', 'P', 'p']:
            mode = simpledialog.askstring(
                "Input",
                "Invalid mode. Choose 'training' or 'production'").lower()

        txt = ""
        # Check if the user has chosen training or production
        if mode == 'T' or mode == 't':
            mode = 'training'
            folder_chosen_path = self.training_folder_start_cam1
            calipers_list = list_subfolders(folder_chosen_path)
            txt += f"Folders already present in {folder_chosen_path}:\n"
            # Insert the text the user can read
            for caliper in calipers_list:
                txt += f"{caliper}\n"
        elif mode == 'P' or mode == 'p':
            mode = 'production'
            folder_chosen_path = self.production_temp_folder_start_cam1
            calipers_list = list_subfolders(folder_chosen_path)

        txt += ("\nIf the caliper is new, please enter a new name.\n"
                "If instead the caliper already exists in the database, "
                "please choose the already existing name.\n")

        # Ask the user to insert the name of the folder
        input_caliper_name = simpledialog.askstring(
            "Enter the folder name where to save the new images", txt)

        # Save the folder path if a name has been inserted
        if input_caliper_name:
            caliper_folder_path = os.path.join(
                folder_chosen_path, input_caliper_name)

        return caliper_folder_path

    def create_temp_folder(self,
                           caliper_folder_path: str) -> None:
        """
        Creates temporary folders for production.
        """
        if not os.path.exists(self.production_temp_folder_start_cam1):
            # Create the temp production folder
            os.mkdir(self.production_temp_folder_start_cam1)
            # Create temp caliper's production folder
            os.mkdir(caliper_folder_path)
            print("Production folder "
                  f"{self.production_temp_folder_start_cam1} created!")
            print(f"Temporary caliper folder "
                  f"{caliper_folder_path} created!")
        else:  # If temp production folder already exists
            if not os.path.exists(caliper_folder_path):
                # Create temp caliper's production folder
                os.mkdir(caliper_folder_path)
            print(f"Production folder {self.production_temp_folder_start_cam1}"
                  " already exists, deleted and created again!")
            print(f"Temporary caliper folder "
                  f"{caliper_folder_path} created!")


class CameraAcquisition(ABC):
    """
    Abstract class for camera acquisition.
    It defines the common interface for all camera acquisition classes.
    """

    def __init__(self,
                 camera_id: int,
                 camera_serial_number: int | str,
                 camera_width: int,
                 camera_height: int) -> None:
        self.camera_id = camera_id
        self.camera_serial_number = camera_serial_number
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame = None
        self.acq_th = None
        self.stop_event = None

    @abstractmethod
    def acquisition_thread(self,
                           stop_event: Event):
        """
        This method defines the acquisition thread target.
        """
        pass

    def start_acquisition_thread(self,
                                 reversed_position: bool):
        """
        Start acquisition thread.
        """
        self.stop_event = Event()
        self.acq_th = Thread(
            target=self.acquisition_thread, args=(self.stop_event,
                                                  reversed_position))
        self.acq_th.start()

    def stop_acquisition_thread(self):
        """
        Stop acquisition thread.
        """
        self.stop_event.set()
        self.acq_th.join()  # Wait for the thread to finish

    def save_frame(self,
                   dest_folder: str):
        """
        Save a new frame to the training/production folder.

        Each image is saved with a progressive number,
        in the form of train_1.bmp, train_2.bmp, etc.
        The suffix of the last saved image is retrieved to save the new image
        with the next progressive number.
        """

        # Get all photos in the destination folder, sorted by creation time
        list_of_files = sorted(
            glob.glob(os.path.join(dest_folder, "train_*.bmp")),
            key=os.path.getctime)
        # Get the last photo path and total number of photos
        if len(list_of_files) > 0:
            last_file = list_of_files[-1]
            number_of_files = int(last_file.split('_')[-1].split('.')[0])
            print(f"Current number of files in {dest_folder} :"
                  f" {number_of_files+1}")
        else:
            number_of_files = 0
        # Set the path of image to be saved
        image_path = os.path.join(
            dest_folder, f"train_{number_of_files + 1}.bmp")
        # Create the destination folder if it doesnt exists
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        # Save image in the destination folder
        cv2.imwrite(image_path, self.frame)

    def get_current_frame(self):
        """
        Return current frame.
        """
        return self.frame


class CameraOpenCV(CameraAcquisition):
    """
    Acquisition class for OpenCV cameras.
    The frame is both captured and displayed
    using the OpenCV library.
    """

    def __init__(self,
                 camera_id: int,
                 camera_serial_number: int | str,
                 camera_width: int,
                 camera_height: int,
                 enable_directshow: bool = False):
        super().__init__(camera_id,
                         camera_serial_number,
                         camera_width,
                         camera_height)
        self.cap = None,
        self.enable_directshow = enable_directshow
        self.frame: np.ndarray = None

    def acquisition_thread(self,
                           stop_event: Event,
                           reversed_position: bool):
        """
        Continuous thread to acquire frames from the camera and display them.

        Args:
        - stop_event (Event): Event object to signal when to stop the thread.
        """
        # Create a VideoCapture object to capture frames from the camera
        if platform.system() == "Windows" and self.enable_directshow:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Failed to open camera")
            return

        # Set the camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        # Start a loop to continuously acquire frames from the camera
        while not stop_event.is_set():
            # Read a frame from the camera
            ret, frame = self.cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Failed to read frame")
                break

            if reversed_position:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Display the frame using OpenCV
            cv2.imshow("Camera", frame)

            # Update current frame
            self.frame = frame

            # Wait for the 'q' key to be pressed to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the VideoCapture object and close the window
        self.cap.release()
        cv2.destroyAllWindows()

    def display_caliper_box(self,
                            bounding_box: tuple[int],
                            camera_width: int,
                            camera_height: int,
                            reversed_position: bool):
        """
        Display a frame from the camera with a rectangle drawn on it.
        The rectangle represents the caliper box, and text instructs the user.

        The function waits for the 'Enter' key to be pressed to exit the loop.
        """

        def resize_image(image, window_width, window_height):
            """
            """
            height, width = image.shape[:2]
            aspect_ratio = width / height

            # Resize
            if window_width * aspect_ratio > window_height:
                new_width = window_height * aspect_ratio
                new_height = window_height
            else:
                new_width = window_width
                new_height = window_width / aspect_ratio

            return int(new_width), int(new_height)

        # Create a VideoCapture object to capture frames from the camera
        if platform.system() == "Windows" and self.enable_directshow:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Failed to open camera")
            return

        # Set the camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        # Start a loop to continuously acquire frames from the camera
        while True:
            # Read a frame from the camera
            ret, frame = self.cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Failed to read frame")
                break

            # Draw a rectangle on the frame with specific coordinates
            top_left = (bounding_box[0], bounding_box[1])
            bottom_right = (bounding_box[2], bounding_box[3])
            color = (0, 255, 0)  # Green color
            thickness = 2
            frame = cv2.rectangle(frame, top_left, bottom_right,
                                  color, thickness)

            # Add text
            text = "Press enter to confirm the position"
            org = (10, 30)  # Top left coordinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 255, 0)  # White text
            thickness = 1
            frame = cv2.putText(frame, text, org, font, font_scale,
                                color, thickness, cv2.LINE_AA)

            # Display the frame using OpenCV
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

            # Get the dimensions
            window_width = 1920
            window_height = 1080

            # Resize image
            resized_frame = cv2.resize(frame,
                                       resize_image(frame,
                                                    window_width,
                                                    window_height))

            if reversed_position:
                resized_frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Mostra l'immagine nel frame "Camera Feed"
            cv2.imshow("Camera Feed", resized_frame)

            # Update current frame
            self.frame = resized_frame

            # Check for key press events
            key = cv2.waitKey(1)

            # Check if 'Enter' key is pressed to exit the loop
            if key == 13:  # 13 is the ASCII code for 'Enter' key
                break

        # Release the VideoCapture object and close the window
        self.cap.release()
        cv2.destroyAllWindows()


class CameraOpenCVHTTP(CameraAcquisition):
    """
    Acquisition class for OpenCV cameras.
    The frame is captured using the OpenCV library
    but is displayed using the FastAPI library.
    """

    def __init__(self,
                 camera_id: int,
                 camera_serial_number: int | str,
                 camera_width: int,
                 camera_height: int,
                 enable_directshow: bool = False):
        super().__init__(camera_id,
                         camera_serial_number,
                         camera_width,
                         camera_height)
        self.cap = None
        self.enable_directshow = enable_directshow
        self.frame: np.ndarray = None

    def acquisition_thread(self,
                           stop_event: Event):
        # Instantiate FastAPI app
        app = FastAPI()

        # Define the video_feed endpoint
        @app.get(f"/video_feed_{self.camera_id}")
        def video_feed():
            return StreamingResponse(
                generate_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame")

        # Define the generate_frames generator.
        # It will be used to continuously stream the frames
        def generate_frames():
            # Open a camera connection using OpenCV
            if platform.system() == "Windows" and self.enable_directshow:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
            # Set the camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            # Check if the camera is opened successfully
            if not self.cap.isOpened():
                print("Failed to open camera")
                return
            # Start a loop to continuously acquire frames from the camera
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                _, jpeg = cv2.imencode('.jpg', frame)
                # Update current frame
                self.frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n\r\n')
            self.cap.release()

        # Run the application on uvicorn web server.
        # The frames will be streamed to the endpoint
        # f"http://localhost:8000/video_feed_{self.camera_id}"
        uvicorn.run(
            app, host="0.0.0.0", port=8000)

# New cameras will be added here eventually...
