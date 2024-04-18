"""
Application entry point.

In absence of a presentation layer / user interface,
this file is used to run the application services
logic by interacting with the corresponding APIs.
"""

import os
import shutil
import time

from application.acquisition import AcquisitionService
from application.production import ProductionService
from application.training import TrainingService
from business.training import TrainingStage, AutoencoderBuildStage


# Set the desired mode of operation:
USAGE = 'PRODUCTION'  # ACQUISITION, TRAINING, PRODUCTION


# Handle the acquisition settings for ACQUISITION:
ACQUISITION_SETUP = {
    'num_images': 25,  # Number of images to be acquired
    # DirectShow API for camera acquisition with OpenCV
    'use_direct_show': True,
    # Either perform the entire real time production cycle
    # (acquisition-preprocessing-inference-visualization)
    # or only acquire images
    'live_production': True,
    'bounding_box': {
        'display': False,
        'coords': (463, 436, 1500, 514)  # (x1, y1, x2, y2)
    }
}

# Handle the build and training settings for TRAINING:
BUILD_SETUP = {
    'optimizer': 'Adam',
    'loss': 'ssim'
}
TRAINING_SETUP = {
    'batch_size': 4,
    'epochs': 500,
    'learning_rate': 0.001,
    'patience': 10,
    'validation_split': 0.2
}

# Handle the preprocessing settings for TRAINING/PRODUCTION:
alignment_template = os.path.join(
    'app', 'data', 'training_photos', 'cam_1_red_caliper',
    'caliper_2', 'train_164_mod.bmp')
PREPROCESSING_SETUP = {
    'grayscale': True,
    'alignment': {
        'activate': True,  # Set False for Training
        'reference_img_path': alignment_template,
        'top_match_perc': 0.8,
        'n_features': 10000,
        'uniform_scaling': True,
    },
    'crop': (390, 558, 369, 1601),  # (y1, y2, x1, x2)
    'blur': (5, 5, 1),  # Odd number
    # resize[0] = crop[3] - crop[2]
    # resize[1] = crop[1] - crop[0]
    'resize': (1232, 168),
    # padding[0] = padding[1] = (resize[0] - resize[1]) / 2
    'padding': (532, 532, 0, 0),
    # === Augmentation ===
    'augmentation': {
        'rotation': {
            'center_list': ([(i/2, i/2) for i in range(1, 5)] +
                            [(1536 - i/2, 2048 - i/2) \
                             for i in range(1, 5)]),
            'selected_centers': 1,
            'angle_tuple': (-0.8, 0.8),
            'selected_angles': 2,
            'scale': 1
        },
        'translation': {
            'x_limit': (-20, 20),
            'y_limit': (-10, 10),
            'num_translations': 3
        },
        'rototranslation': True,
        'contrast_brightness': {
            'num_start_images': 8,
            'num_transformations': 1,
            'contrast_factor': 0.2,
            'brightness_factor': (-0.05, 0.08)
        },
    }
}

# Handle the visualization settings for PRODUCTION:
DEFECTS_VISUALIZATION = {
    'plot_defects_over_ssim_map': True,
    'plot_defects_over_original_img': False,
    'ssim_map': {
        'smoothing': [0.01, 0.03],
        'max_luminance': 255,
        'kernel_size': 11,
        'kernel_stdev': 1.5
    },
    'ssim_threshold': 225,
    'min_cluster_size': 15,
    'min_perc_appearance': 40,
    'max_distance_xy': 5,  # For overall clusters computation
    # -- eps: The maximum distance between two samples for one
    # to be considered as in the neighborhood of the other.
    # -- min_samples: The number of samples in a neighborhood
    # for a point to be considered as a core point, including the point itself.
    'dbscan_params': (13, 1),  # (eps, min_samples)
    # Use this to better visualize the image plotted by matplotlib
    'area_to_plot': ((0, 0), (192, 192)),  # Remove pixels cropped
    'visualize_real_time': True,  # To show plots in real time
    # ! Make sure the .txt files of the defective coordinates
    # are in the subfolders before running this search
    'find_tp_fp_fn': False,  # To perform the search of TP, FP, FN
    'store_insights_to_excel': False,
    'exclude_reconstruction_errors': True
}


# Run the desired mode.
if USAGE == 'ACQUISITION':

    # Open an acquisition service (for each camera)
    acquisition_cam_1 = AcquisitionService(camera_type="OpenCV")

    # Define the training and the production folders
    # where to save the new acquisitions
    training_folder_start_cam1 = os.path.join(
        "app", "data", "training_photos", "cam_1_red_caliper")
    production_folder_start_cam1 = os.path.join(
        "app", "data", "production_photos", "cam_1_red_caliper")
    production_temp_folder_start_cam1 = os.path.join(
        "app", "data", "production_temp_photos", "cam_1_red_caliper")

    production_temp_folder = os.path.join(
        "app", "data", "production_temp_photos")
    weights_temp_folder = os.path.join(
        "app", "data", "weights_temp")

    if os.path.exists(production_temp_folder):
        shutil.rmtree(production_temp_folder)
    os.makedirs(production_temp_folder)

    if os.path.exists(weights_temp_folder):
        shutil.rmtree(weights_temp_folder)
    os.makedirs(weights_temp_folder)

    # Get the name of the selected caliper folder
    caliper_folder = acquisition_cam_1.manage_acquisition(
        training_folder_start_cam1,
        production_temp_folder_start_cam1)

    # Open camera(s) connection(s)
    acquisition_cam_1.open_camera_connection(
        camera_id=0,
        camera_serial_number="0",
        camera_width=2048,
        camera_height=1536,
        enable_directshow=ACQUISITION_SETUP["use_direct_show"],
        reversed_position=True,  # If the caliper is reversed
        bounding_box=ACQUISITION_SETUP["bounding_box"])

    WARMUP_TIME = 5
    # Before saving images, wait for the camera to open.
    time.sleep(WARMUP_TIME)

    # Use the acquisition service to save images.
    # The images will be saved in the above training folder(s).
    # If the frames are streamed to a web server using a camera_type like
    # "OpenCVHTTP" in the acquisition service (for compatibility
    # with environments which do not provide a GUI, like a Docker container)
    # you'll have to start the frames streaming with a get request
    # at f"http://localhost:8000/video_feed_{camera_id}"
    # and handle the error if the get request
    # is not sent before a frame request
    t0 = time.time()
    while True:
        # Save the acquired image in the destination caliper's folder
        acquisition_cam_1.save_image(dest_folder=caliper_folder)
        time.sleep(1)
        # Decide when to close the connection
        if time.time() - t0 > ACQUISITION_SETUP["num_images"]:
            acquisition_cam_1.close_camera_connection()
            break

    # Real time production cycle:
    if ACQUISITION_SETUP["live_production"] and 'production' in caliper_folder:

        # Start the production stage on that temp caliper's folder
        production_temp_folder_cam1 = os.path.join(
            "app", "data", "production_temp_photos", "cam_1_production")
        # Define the temp weights folder
        weights_temp_folder_cam1 = os.path.join("app", "data", "weights_temp",
                                                "temp_cam_1")
        if not os.path.exists(weights_temp_folder_cam1):
            os.makedirs(weights_temp_folder_cam1)
        # Copy the last weights.h5 file from the original folder to temp folder
        weights_folder_cam1 = os.path.join("app", "data", "weights", "cam_1")
        weights_h5_path = os.path.join(weights_folder_cam1, "weights.h5")
        shutil.copy(weights_h5_path, weights_temp_folder_cam1)

        # Retrieve the initial network architecture
        training_stage_cam1 = TrainingStage(
            build_stage=AutoencoderBuildStage(
                build_setup=None,
                training_setup=None,
                weights_folder=weights_temp_folder_cam1,
                preprocessing_setup=PREPROCESSING_SETUP),
            train_stage=None
        )
        training_stage_cam1.set_initial_network()
        initial_network = training_stage_cam1.initial_network

        # Open a production service (for each camera)
        production_cam_1 = ProductionService(
            production_start_folder=production_temp_folder_start_cam1,
            production_folder=production_temp_folder_cam1,
            weights_folder=weights_temp_folder_cam1,
            camera=acquisition_cam_1.camera,
            application='Defect Detection',
            initial_network=initial_network,
            preprocessing_setup=PREPROCESSING_SETUP,
            processing_library='OpenCV'
        )

        # Get the inference results
        print("A request for inference has been received...")

        # Preprocessing
        t0_cpu = time.process_time()
        t0_real = time.perf_counter()
        production_cam_1.features_1.process()
        print("CPU time for preprocessing: "
              f"{round(time.process_time()-t0_cpu, 3)} s")
        print("Real time for preprocessing: "
              f"{round(time.perf_counter()-t0_real, 3)} s")
        # Inference
        t0_cpu = time.process_time()
        t0_real = time.perf_counter()
        production_cam_1.features_2.handle_inference_results()
        print("CPU time for inference: "
              f"{round(time.process_time()-t0_cpu, 3)} s")
        print("Real time for inference: "
              f"{round(time.perf_counter()-t0_real, 3)} s")
        # Visualization
        production_cam_1.features_2.handle_overall_visualization(
            DEFECTS_VISUALIZATION)

elif USAGE == 'TRAINING':

    training_folder_start_cam1 = os.path.join(
        "app", "data", "training_photos", "cam_1_red_caliper")

    training_folder_cam1 = os.path.join(
        "app", "data", "training_photos", "cam_1_training")

    # Define the weights folder(s) where the trained weights will be saved
    # for each camera
    weights_folder_cam1 = os.path.join("app", "data", "weights", "cam_1")

    # Open a training service (for each camera)
    training_service_cam1 = TrainingService(
        training_start_folder=training_folder_start_cam1,
        training_folder=training_folder_cam1,
        weights_folder=weights_folder_cam1,
    )

    training_service_cam1.start_training_pipeline(
        application='Defect Detection',
        preprocessing_setup=PREPROCESSING_SETUP,
        processing_library='OpenCV',
        build_setup=BUILD_SETUP,
        training_setup=TRAINING_SETUP
    )

elif USAGE == 'PRODUCTION':

    production_folder_start_cam1 = os.path.join(
        "app", "data", "production_photos", "cam_1_red_caliper")

    production_folder_cam1 = os.path.join(
        "app", "data", "production_photos", "cam_1_production")

    # Define the weights folder(s) where the trained weights are saved
    weights_folder_cam1 = os.path.join("app", "data", "weights", "cam_1")

    # Retrieve the initial network architecture
    training_stage_cam1 = TrainingStage(
        build_stage=AutoencoderBuildStage(
            build_setup=None,
            training_setup=None,
            weights_folder=weights_folder_cam1,
            preprocessing_setup=PREPROCESSING_SETUP),
        train_stage=None
    )
    training_stage_cam1.set_initial_network()
    initial_network = training_stage_cam1.initial_network

    # Open an acquisition service (for each camera)
    acquisition_cam_1 = AcquisitionService(camera_type="OpenCV")

    # Open a production service (for each camera)
    production_cam_1 = ProductionService(
        production_start_folder=production_folder_start_cam1,
        production_folder=production_folder_cam1,
        weights_folder=weights_folder_cam1,
        camera=acquisition_cam_1.camera,
        application='Defect Detection',
        initial_network=initial_network,
        preprocessing_setup=PREPROCESSING_SETUP,
        processing_library='OpenCV'
    )

    # Get the inference results
    print("A request for inference has been received...")

    """
    # Preprocessing
    t0_cpu = time.process_time()
    t0_real = time.perf_counter()
    production_cam_1.features_1.process()
    print("CPU time for preprocessing: "
          f"{round(time.process_time()-t0_cpu, 3)} s")
    print("Real time for preprocessing: "
          f"{round(time.perf_counter()-t0_real, 3)} s")
    # Inference
    t0_cpu = time.process_time()
    t0_real = time.perf_counter()
    production_cam_1.features_2.handle_inference_results()
    print("CPU time for inference: "
          f"{round(time.process_time()-t0_cpu, 3)} s")
    print("Real time for inference: "
          f"{round(time.perf_counter()-t0_real, 3)} s")
    """
    # Visualization
    production_cam_1.features_2.handle_overall_visualization(
        DEFECTS_VISUALIZATION)
