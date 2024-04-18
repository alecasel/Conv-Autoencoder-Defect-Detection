"""
Business logic for production component.
"""

import os
import shutil
import time
from typing import List, Optional, Set, Tuple
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import Any
from sklearn.cluster import DBSCAN
import keras

from business.training import AutoencoderBuildStage, TrainingStage
from shared.common import (FramePreprocessorOpenCV,
                           verify_excel_sheet_existence,
                           verify_excel_file_existence)
from business.acquisition import CameraAcquisition


class PreprocessingFeatures():
    """
    This class defines the preprocessing stage.
    """

    def __init__(self,
                 camera: CameraAcquisition,
                 production_start_folder: str,
                 production_folder: str,
                 preprocessing_setup: dict[str, Any] | None,
                 processing_library: str) -> None:
        self.camera = camera
        self.production_start_folder = production_start_folder
        self.production_folder = production_folder
        self.preprocessing_setup = preprocessing_setup
        self.processing_library = processing_library

    def process(self):
        """
        Apply preprocessing to the production
        images inside production_start_folder.
        The preprocessing setup is defined in preprocessing_setup dictionary.
        """
        try:
            if os.path.exists(self.production_folder):
                # Remove the already existing folder
                shutil.rmtree(self.production_folder)
            # Create production folder
            os.mkdir(self.production_folder)
            for subf in os.listdir(self.production_start_folder):
                src = os.path.join(self.production_start_folder, subf)
                dst = os.path.join(self.production_folder, subf)
                # Create the empty caliper subfolder if does not exist
                if not os.path.isdir(dst):
                    shutil.copytree(src, dst)
            print("Production folder "
                  f"{self.production_folder} created!")

        except OSError as e:
            print(f"Error: {self.production_folder} not copied. {e}")

        if self.processing_library is None:
            print('No preprocessing library has been specified, '
                  'skipping preprocessing...')
        else:
            # Apply preprocessing to all images in the production folder
            if self.processing_library == "OpenCV":

                # Store the template reference for 2d matching
                template_img = None
                alignment_setup = self.preprocessing_setup["alignment"]
                if alignment_setup:
                    img_path = alignment_setup["reference_img_path"]
                    template_img = cv2.cvtColor(cv2.imread(img_path),
                                                cv2.COLOR_BGR2GRAY)

                print('Using OpenCV for preprocessing images in'
                      f' {self.production_folder}')

                for subf in os.listdir(self.production_folder):
                    subf_path = os.path.join(self.production_folder, subf)
                    for filename in os.listdir(subf_path):
                        file_path = os.path.join(subf_path, filename)
                        t0_cpu = time.process_time()
                        t0_real = time.perf_counter()
                        if filename.endswith(".bmp"):
                            image = cv2.imread(file_path)
                            # Apply preprocessing
                            image = FramePreprocessorOpenCV().preprocess_frame(
                                frame=image,
                                preprocessing_setup=self.preprocessing_setup,
                                template_img=template_img)
                            cv2.imwrite(file_path, image)
                            print(f"CPU time for preprocessing {filename}: "
                                  f"{round(time.process_time()-t0_cpu, 3)} s")
                            print(f"Real time for preprocessing {filename}: "
                                  f"{round(time.perf_counter()-t0_real, 3)} s")


class AutoencoderFeatures():

    """
    This class implements the production features for the autoencoder model.
    """

    def __init__(self,
                 production_folder: str,
                 weights_folder: str,
                 initial_network: str) -> None:
        self.production_folder = production_folder
        self.weights_folder = weights_folder
        self.initial_network = initial_network

    def get_initial_network(self) -> keras.Model:
        """
        Load pre-trained weights for the autoencoder model.

        Returns:
            keras.Model: Autoencoder model with loaded weights.
        """
        autoencoder = self.initial_network
        weights_path = os.path.join(self.weights_folder, "weights.h5")
        autoencoder.load_weights(filepath=weights_path)
        return autoencoder

    def perform_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Perform inference using the autoencoder model.

        Parameters:
            frame (np.ndarray): Input image for inference.

        Returns:
            np.ndarray: Reconstructed image after inference.
        """
        # Normalize the image data (as in training)
        x_test = frame.astype('float32') / 255.0
        # Add batch dimension (a dimension along axis=0)
        x_test = np.expand_dims(x_test, axis=0)
        # Store the autoencoder
        autoencoder = self.get_initial_network()
        # Do the inference
        reconstructed_frame = autoencoder.predict(x_test)
        reconstructed_frame = reconstructed_frame[0] * 255.0

        return reconstructed_frame

    def handle_inference_results(self):
        """
        Perform inference on each BMP file in the production folder,
        save original and reconstructed frames in weights subfolders,
        and copy ground truth coordinate files
        to corresponding weights subfolders.
        """
        # Iterate through subfolders in the production folder
        for subfolder in os.listdir(self.production_folder):
            production_subfolder_path = os.path.join(self.production_folder,
                                                     subfolder)
            weights_subfolder_path = os.path.join(self.weights_folder,
                                                  subfolder)

            # Create an empty weights subfolder if it doesn't exist
            if not os.path.isdir(weights_subfolder_path):
                os.mkdir(weights_subfolder_path)

            # Process each file in the production subfolder
            for file in os.listdir(production_subfolder_path):
                file_path = os.path.join(production_subfolder_path, file)

                if file.endswith(".bmp"):
                    # Perform inference on BMP files
                    original_frame = cv2.cvtColor(cv2.imread(file_path),
                                                  cv2.COLOR_BGR2GRAY)
                    print(f"\nPerforming inference on {file}...")
                    t0_cpu = time.process_time()
                    t0_real = time.perf_counter()
                    reconstructed_frame = self.perform_inference(
                        original_frame)
                    t1_cpu = time.process_time()
                    t1_real = time.perf_counter()
                    print(f"CPU Inference time: {round(t1_cpu-t0_cpu, 3)} s.")
                    print("Real Inference time: "
                          f"{round(t1_real-t0_real, 3)} s.")

                    # Save original and reconstructed frames
                    # in the weights subfolder
                    original_path_to_save = os.path.join(
                        weights_subfolder_path, file[:-4] + "_A_original.bmp")
                    cv2.imwrite(original_path_to_save, original_frame)

                    reconstructed_path_to_save = os.path.join(
                        weights_subfolder_path,
                        file[:-4] + "_B_reconstructed.bmp")
                    cv2.imwrite(reconstructed_path_to_save,
                                reconstructed_frame)

                elif file.endswith(".txt"):
                    # Copy ground truth coordinate files
                    # to the weights subfolder
                    shutil.copy(file_path, weights_subfolder_path)

    def compute_ssim(self,
                     original_frame: np.ndarray,
                     reconstructed_frame: np.ndarray,
                     smoothing: tuple,
                     max_luminance: int,
                     kernel_size: int,
                     kernel_stdev: float) -> np.ndarray:
        """
        Compute Structural Similarity Index (SSIM) between two images.

        Parameters:
            original_frame (np.ndarray): Original image.
            reconstructed_frame (np.ndarray): Reconstructed image.
            smoothing (tuple): Smoothing constants for SSIM calculation.
            Should be in the range [0.01, 0.03].
            max_luminance (int): Maximum level of luminance
            in grayscale images, typically 255.
            kernel_size (int): Size of the Gaussian kernel for convolution.
            kernel_stdev (float): Standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: SSIM map representing the local SSIM values
            across the images.
        """
        # Constants to ensure numerical stability in SSIM
        C1 = (smoothing[0] * max_luminance)**2
        C2 = (smoothing[1] * max_luminance)**2

        # Gaussian bidimensional kernel
        kernel = cv2.getGaussianKernel(kernel_size, kernel_stdev)
        # Convolutional window
        window = np.outer(kernel, kernel.T)

        # Convert input images to float64
        original_frame = np.float64(original_frame)
        reconstructed_frame = np.float64(reconstructed_frame)

        # Compute local means
        mu1 = cv2.filter2D(original_frame, -1, window,
                           borderType=cv2.BORDER_WRAP)
        mu2 = cv2.filter2D(reconstructed_frame, -1, window,
                           borderType=cv2.BORDER_WRAP)

        # Squared means and mean of products
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        # Local variances
        sigma1_sq = cv2.filter2D(
            original_frame * original_frame, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(
            reconstructed_frame * reconstructed_frame, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(
            original_frame * reconstructed_frame, -1, window) - mu1_mu2

        # Calculate SSIM map
        ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

        valid_size = (np.array(original_frame.shape) - np.array(window.shape)
                      + 1)
        pad = (np.array(original_frame.shape) - valid_size) // 2
        ssim_map = ssim_map[pad[0]:-pad[0], pad[1]:-pad[1]]

        return ssim_map

    def get_defects_mask(
            self,
            ssim_map: np.ndarray,
            ssim_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a mask assigning 'True' values (defects)
        where similarity < ssim_threshold.

        Parameters:
            ssim_map (np.ndarray): Structural Similarity Index (SSIM) map.
            ssim_threshold (float): Threshold for identifying defects.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing the defects mask
            and coordinates of defective pixels.
        """
        # Convert similarity map from [0,1] to [0,255]
        ssim_map_abs = abs(ssim_map) * 255
        # Create a mask assigning "True" values where similarity is low
        mask_defects_ssim_thresh = ssim_map_abs < ssim_threshold
        # Get all the coordinates of defective pixels
        clusters_coords_ssim_thresh = np.argwhere(mask_defects_ssim_thresh)

        return mask_defects_ssim_thresh, clusters_coords_ssim_thresh

    def draw_defective_pixels(
        self,
        original_frame: np.ndarray,
        ssim_map: np.ndarray,
        mask_defects_ssim_thresh: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw defective pixels on the original and SSIM images.

        Parameters:
            original_frame (np.ndarray): Original grayscale image.
            ssim_map (np.ndarray): Structural Similarity Index (SSIM) map.
            mask_defects_ssim_thresh (np.ndarray): Defects mask.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Tuple containing images with defective pixels drawn.
        """
        # Create a copy of the similarity map with black defective pixels
        black_defects_img = np.copy(abs(ssim_map) * 255)
        # Black set to defective pixels
        black_defects_img[mask_defects_ssim_thresh] = 0
        # Convert the grayscale image to a 3-channel image (RGB)
        black_defects_img = np.repeat(
            black_defects_img[:, :, np.newaxis], 3, axis=2)
        # Normalizes data in range 0 - 255
        black_defects_img = black_defects_img / black_defects_img.max()
        black_defects_img = 255 * black_defects_img
        black_defects_img = black_defects_img.astype(np.uint8)

        # Create an image with red defective pixels
        red_defects_img = np.zeros_like(ssim_map, dtype=np.uint8)
        # Red channel set to defective pixels
        red_defects_img[mask_defects_ssim_thresh] = 255
        # Convert the grayscale image to a 3-channel image (RGB)
        # where Red is the only active channel
        red_defects_img = np.repeat(
            red_defects_img[:, :, np.newaxis], 3, axis=2)
        red_defects_img[..., 1:] = 0  # Green and Blue channels set to 0
        # Convert RGB to BGR for OpenCV
        # red_defects_img = red_defects_img[..., ::-1]

        # Combine: original image (1088,1088,1) --> crop top and left +
        # + red defective pixels over a threshold (1078,1078,1).
        # Compute the difference in shape
        # between the original frame and the ssim map
        delta_shape = original_frame.shape[0] - red_defects_img.shape[0]
        # Convert from grayscale to BGR
        # Crop the original frame to equal its shape to ssim map's shape
        original_reshaped_img = original_frame[
            int(delta_shape/2):original_frame.shape[0]-int(delta_shape/2),
            int(delta_shape/2):original_frame.shape[0]-int(delta_shape/2)]
        # Convert the grayscale image to a 3-channel image (RGB)
        original_reshaped_img = np.stack([original_reshaped_img] * 3, axis=-1)
        # Create a copy of 'original_reshaped_img'
        original_red_defects = original_reshaped_img.copy()
        # Set Red for defective pixels
        original_red_defects[mask_defects_ssim_thresh, 0] = 255
        # Set Green and Blue to 0
        original_red_defects[mask_defects_ssim_thresh, 1:] = 0
        # Convert RGB to BGR for OpenCV
        # original_red_defects = original_red_defects[..., ::-1]

        # Combine: ssim with black defective pixels (1078,1078,1) +
        # + red defective pixels over a threshold (1078,1078,1).
        ssim_red_defects = black_defects_img + red_defects_img

        return (original_red_defects, ssim_red_defects,
                black_defects_img, original_reshaped_img)

    def get_defective_clusters(self,
                               dbscan_params: tuple[float, int],
                               clusters_coords_ssim_thresh: np.ndarray) \
            -> dict[tuple: (int, list, (int, int), str)]:
        """
        """

        def get_centroid(points):
            """
            """
            mean_y = round(np.mean(points[:, 0]))
            mean_x = round(np.mean(points[:, 1]))
            return (mean_x, mean_y)

        def get_defect_shape(cluster_points):
            """
            """
            x_values = []  # x coordinates of the points of the cluster
            y_values = []  # y coordinates of the points of the cluster
            for point in cluster_points:
                x_values.append(point[1])
                y_values.append(point[0])

            # Compute the top left and bottom right points of the box
            top_left_vertex = (min(x_values), min(y_values))
            bottom_right_vertex = (max(x_values), max(y_values))
            # Compute the width, the height and the area of the box
            box_width = bottom_right_vertex[0] - top_left_vertex[0]
            box_height = bottom_right_vertex[1] - top_left_vertex[1]
            box_area = box_width * box_height

            return box_width, box_height, box_area

        def is_reconstruction_error(box_width, box_height):
            """
            """
            if ((box_width < 10 and box_height > 49)
                    or (box_width > 49 and box_height < 10)
                    or (box_width < 4 and box_height > 15)
                    or (box_height < 4 and box_width > 15)
                    or (box_height < 3) or (box_width < 3)):
                return True
            else:
                return False

        # Unpack DBSCAN parameters
        max_distance_neigh, min_neigh_samples_core = dbscan_params
        # Apply DBSCAN clustering algorithm
        dbscan = DBSCAN(eps=max_distance_neigh,
                        min_samples=min_neigh_samples_core,
                        metric='euclidean')

        # Associate each point to a cluster label
        clusters_labels = dbscan.fit_predict(clusters_coords_ssim_thresh)
        # Get how many times the cluster labels are found
        unique_clusters_labels, cluster_size = np.unique(
            clusters_labels, return_counts=True)

        # Put each cluster label together with its points
        clusters = {}
        for cluster_label, size in zip(unique_clusters_labels,
                                       cluster_size):
            # Group all the clusters with the same label together
            clusters_mask = (clusters_labels == cluster_label)
            # Get the coordinates of all the points with the same label
            cluster_points = clusters_coords_ssim_thresh[clusters_mask]
            # Get the centroid of the cluster
            centroid = get_centroid(cluster_points)
            # Get the defect's shape
            width, height, area = get_defect_shape(cluster_points)
            if is_reconstruction_error(width, height):
                reconstruction_flag = 'Reconstruction Error'
            else:
                reconstruction_flag = 'Real Defect'
            # Store the cluster
            clusters[centroid] = (size, cluster_points,
                                  (width, height), reconstruction_flag)

        return clusters

    def count_big_clusters(
            self,
            min_cluster_size: int,
            clusters: dict[tuple: (int, list, (int, int), str)]) -> int:
        """
        Returns the number of big clusters found in an acquisition.
        """
        big_clusters_count = 0
        for count, _, _, _ in clusters.values():
            if count >= min_cluster_size:
                big_clusters_count += 1

        return big_clusters_count

    def get_overall_centroids(
        self,
        min_cluster_size: int,
        exclude_reconstruction_errors: bool,
        overall_centroids: dict[tuple],
        new_clusters: dict[tuple: (int, list, (int, int), str)],
        max_distance_xy: int = 5) \
            -> dict[tuple, float]:
        """
        Return the centroids across all the images acquired for each caliper.
        The percentage of detection is stored.
        """

        def is_similar_cluster(coords_1: Tuple[int, int],
                               coords_2: Tuple[int, int],
                               max_distance_xy) -> bool:
            """
            Return True if the two coordinates belong to the same cluster
            because their difference is not superior to 'max_distance_xy';
            otherwise return False.
            """
            return (abs(coords_1[0] - coords_2[0]) <= max_distance_xy
                    and abs(coords_1[1] - coords_2[1]) <= max_distance_xy)

        def get_most_similar_cluster(coords_new: Tuple[int, int],
                                     list_coords_old: List[Tuple[int, int]]) \
                -> Tuple[int, int]:
            """
            Return the coordinates of the centroid of the most similar cluster
            to the new cluster, chosen among the list of the already existing
            clusters.
            """
            # Convert coordinates into numpy arrays
            new_c = np.array(coords_new)
            distances = []  # Distances between each couple of centroids
            for coords_old in list_coords_old:
                old_c = np.array(coords_old)
                distances.append(np.linalg.norm(old_c - new_c))
            # Find the index of the min distance
            min_index = distances.index(min(distances))
            # Get the old_c associated with the min distance
            return list_coords_old[min_index]

        # Filter the new centroids by big clusters and reconstruction defects
        single_centroids = list(new_clusters.keys())
        for centroid, (size, _, _, flag) in new_clusters.items():
            if (size < min_cluster_size
                    and centroid in single_centroids):  # If not removed yet
                single_centroids.remove(centroid)  # Remove the little clusters
            if (exclude_reconstruction_errors
                    and flag == 'Reconstruction Error'
                    and centroid in single_centroids):
                single_centroids.remove(centroid)  # Remove rec errors

        # Store the centroids of the clusters of the already passed images
        old_centroids = list(overall_centroids.keys())

        for new_c in single_centroids:  # Process all the new centroids
            if len(old_centroids) == 0:  # First time the function is called
                # Add the new cluster to the overall dict and count 1
                overall_centroids[new_c] = 1
            else:  # In the overall dict there are already clusters
                similar_old_c = []  # The old centroids similar to the new one
                # Check if new_c is already into the overall dict
                for old_c in old_centroids:
                    if is_similar_cluster(new_c, old_c, max_distance_xy):
                        similar_old_c.append(old_c)  # Add old_c
                # Manage the comparison cases
                if len(similar_old_c) == 0:  # Found no similar centroids
                    overall_centroids[new_c] = 1
                elif len(similar_old_c) == 1:  # Found one similar centroid
                    overall_centroids[similar_old_c[0]] += 1
                elif len(similar_old_c) > 1:  # Found more than one similar
                    # Check which one is the most similar
                    old_c = get_most_similar_cluster(new_c, similar_old_c)
                    overall_centroids[old_c] += 1

        return overall_centroids

    def plot_defective_pixels(
            self,
            subfolder_weights: str,
            bmp_filename: str,
            area_to_plot: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
            ssim_threshold: int,
            original_red_defects: np.ndarray,
            ssim_red_defects: np.ndarray,
            plot_flag: List[bool]) -> None:
        """
        Plot defective pixels and save the images based on the specified flags.

        Parameters:
            subfolder_weights (str): Subfolder path for saving the images.
            bmp_filename (str): BMP filename for saving the images.
            area_to_plot (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            Coordinates specifying the area to plot.
            ssim_threshold (int): Similarity threshold value.
            original_red_defects (np.ndarray): Image with red defective pixels.
            ssim_red_defects (np.ndarray):
            Image with red defective pixels based on SSIM.
            plot_flag (List[bool]): Flags indicating whether to plot
            original_red_defects and ssim_red_defects.
        """

        if not plot_flag[0] and not plot_flag[1]:
            return

        if area_to_plot is not None:
            start_coord = area_to_plot[0]
            end_coord = area_to_plot[1]

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      label='Defective region',
                                      markerfacecolor='red', markersize=5)]

        if plot_flag[0]:
            # Plot 'original_red_defects'
            # with description "train_1_C_thresh_200.png"
            path_to_save = os.path.join(subfolder_weights,
                                        bmp_filename +
                                        f"_C_thresh_{ssim_threshold}.png")
            if area_to_plot is not None:
                # Select area of interest to plot
                original_red_defects = original_red_defects[
                    start_coord[1]:end_coord[1],
                    start_coord[0]:end_coord[0]]

            plt.title(f"Similarity threshold = {ssim_threshold}/255")
            plt.imshow(original_red_defects, cmap='gray')
            ax = plt.subplot(111)
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
            # Put a legend below the current axis
            ax.legend(handles=legend_elements,
                      loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=False, ncol=1)
            plt.axis('off')
            plt.savefig(path_to_save, format='png', dpi=250)
            plt.close()

        if plot_flag[1]:
            # Plot 'ssim_red_defects'
            # with description "train_1_C_ssim_thresh_200.png"
            path_to_save = os.path.join(subfolder_weights,
                                        bmp_filename +
                                        f"_C_ssim_thresh_{ssim_threshold}.png")
            # Select area of interest to plot
            if area_to_plot is not None:
                ssim_red_defects = ssim_red_defects[
                    start_coord[1]:end_coord[1],
                    start_coord[0]:end_coord[0]]

            plt.title(f"Similarity threshold = {ssim_threshold}/255")
            plt.imshow(ssim_red_defects, cmap='gray')
            ax = plt.subplot(111)
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
            # Put a legend below the current axis
            ax.legend(handles=legend_elements,
                      loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=False, ncol=1)
            plt.axis('off')
            plt.savefig(path_to_save, format='png', dpi=250)
            plt.close()

    def plot_big_clusters(
            self,
            subfolder_weights: str,
            bmp_filename: str,
            area_to_plot: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
            ssim_threshold: int,
            min_cluster_size: int,
            black_defects_img: np.ndarray,
            original_reshaped_img: np.ndarray,
            clusters: dict[tuple: (int, list, (int, int), str)],
            plot_flag: List[bool]) -> None:
        """
        Plot defective clusters and save the images
        based on the specified flags.

        Parameters:
            subfolder_weights (str): Subfolder path for saving the images.
            bmp_filename (str): BMP filename for saving the images.
            area_to_plot (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            Coordinates specifying the area to plot.
            ssim_threshold (int): Similarity threshold value.
            min_cluster_size (int): Minimum cluster size.
            black_defects_img (np.ndarray): Image with black defective pixels.
            original_reshaped_img (np.ndarray): Original reshaped image.
            big_clusters_sample_counts (dict):
            Dictionary with cluster labels and associated data.
            plot_flag (List[bool]): Flags indicating whether to plot
            black_defects_clusters_img and original_reshaped.
        """
        if not plot_flag[0] and not plot_flag[1]:
            return

        # Initialize the image where to draw clusters
        accumulated_circles_img = np.zeros_like(black_defects_img)

        cmap = plt.get_cmap("viridis")  # Select colors from this map
        num_big_clusters = self.count_big_clusters(min_cluster_size, clusters)
        # Generate as many colors as the clusters
        colors = cmap(np.linspace(0, 1, num_big_clusters))[:, :3] * 255
        color_index = 0
        # Plot all the points of each cluster with the same color
        # and assign a different color to each cluster
        for size, points, _, _ in clusters.values():
            if size >= min_cluster_size:  # Plot only the big clusters
                color = colors[color_index]
                color_index += 1
                for point in points:  # Color the same all the points
                    # Draw a circle for each point belonging to the cluster
                    accumulated_circles_img = cv2.circle(
                        img=accumulated_circles_img,
                        center=(point[1], point[0]),  # Reversed for DBSCAN
                        radius=4, color=color[::-1],  # From RGB to BGR
                        thickness=-1)  # Fill the circle

        # Create a mask assigning 'True'
        # to all colored pixels of 'accumulated_circles_img'
        mask_clusters = np.any(accumulated_circles_img != 0, axis=-1)

        if plot_flag[1]:  # SSIM map as background
            # Create a copy to avoid overwriting
            black_defects_clusters_img = black_defects_img.copy()
            black_defects_clusters_img[mask_clusters] = (
                accumulated_circles_img[mask_clusters])
            # Plot 'black_defects_clusters_img'
            path_to_save = os.path.join(
                subfolder_weights,
                bmp_filename +
                f"_C_ssim_thresh_{ssim_threshold}" +
                f"_min_cluster_sz_{min_cluster_size}.png")
            if area_to_plot is not None:
                start_coord = area_to_plot[0]
                end_coord = area_to_plot[1]
                black_defects_clusters_img = black_defects_clusters_img[
                    start_coord[1]:end_coord[1], start_coord[0]:end_coord[0]]
            plt.title(f"Similarity Threshold = {ssim_threshold}/255, "
                      f"Min Cluster Size = {min_cluster_size}")
            plt.imshow(black_defects_clusters_img, cmap='gray')
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
            plt.axis('off')
            plt.savefig(path_to_save, format='png', dpi=250)
            plt.close()

        if plot_flag[0]:  # Original image as background
            # Create a copy to avoid overwriting
            original_reshaped = np.copy(original_reshaped_img)
            original_reshaped[mask_clusters] = (
                accumulated_circles_img[mask_clusters])
            # Plot 'original_reshaped'
            path_to_save = os.path.join(
                subfolder_weights,
                bmp_filename +
                f"_C_thresh_{ssim_threshold}" +
                f"_min_cluster_sz_{min_cluster_size}.png")
            if area_to_plot is not None:
                start_coord = area_to_plot[0]
                end_coord = area_to_plot[1]
                original_reshaped = original_reshaped[
                    start_coord[1]:end_coord[1], start_coord[0]:end_coord[0]]
            plt.title(f"Similarity Threshold = {ssim_threshold}/255, "
                      f"Min Cluster Size = {min_cluster_size}")
            plt.imshow(original_reshaped, cmap='gray')
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
            plt.axis('off')
            plt.savefig(path_to_save, format='png', dpi=250)
            plt.close()

    def plot_overall_defective_clusters(
            self,
            caliper_folder_path: str,
            caliper_folder_name: str,
            images_per_caliper: int,
            area_to_plot: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
            ssim_threshold: int,
            min_cluster_size: int,
            black_defects_img: np.ndarray,
            original_reshaped_img: np.ndarray,
            overall_centroids: dict[tuple: float],
            min_perc_appearance: float,
            visualize_real_time: bool) -> None:
        """
        Plot overall defective clusters and save the image.

        Parameters:
            - subfolder_weights (str): Subfolder path for saving the image.
            - caliper_folder (str): Caliper folder name.
            - images_per_caliper (int): Number of images per caliper.
            - area_to_plot (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            Coordinates specifying the area to plot.
            - ssim_threshold (int): SSIM threshold value.
            - min_cluster_size (int): Minimum cluster size.
            - black_defects_img (np.ndarray):
            Image with black defective pixels.
            - original_reshaped_img (np.ndarray): Original reshaped image.
            - cluster_count_dict (Dict[Tuple[int, int],
            - Tuple[float, Tuple[int, int, int]]]): Dictionary with cluster
            coordinates as keys and associated data as values.
            - min_perc_appearance (float): Minimum percentage appearance
            for a cluster to be plotted.
            - visualize_real_time (bool): Flag indicating whether
            to visualize the image in real time.
        """
        def choose_text_color(
            background_color: Tuple[int, int, int]) \
                -> Tuple[int, int, int]:
            """
            Choose the text color based on the background color.

            Parameters:
                background_color (Tuple[int, int, int]): Background color.

            Returns:
                Tuple[int, int, int]: Text color.
            """
            brightness = np.sqrt(
                0.299 * background_color[0]**2 +
                0.587 * background_color[1]**2 +
                0.114 * background_color[2]**2)

            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            return text_color

        # Generate colors for the clusters
        cmap = plt.get_cmap("viridis")
        # Choose as many colors as the number of overall centroids
        num_colors = 0
        for perc in overall_centroids.values():
            if perc >= min_perc_appearance:
                num_colors += 1
        colors = cmap(np.linspace(0, 1, num_colors))[:, :3] * 255
        side_length = 15  # Assign the length to the defective squares
        # Define the font characteristics to plot the numbers
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        legend_elements = []  # Initialize the legend elements
        cluster_num = 0  # Initialize the number of the defect
        # Initialize the image with accumulated squares
        accumulated_squares_img = np.zeros_like(black_defects_img)
        color_index = 0
        for centroid, perc in overall_centroids.items():
            cluster_num += 1  # Increase the number for the defect
            # Check the percentage of detection of the centroid
            if perc >= min_perc_appearance:
                color = colors[color_index]
                color_index += 1
                # 1- Define the verteces of the square
                top_left = (centroid[0] - side_length // 2,
                            centroid[1] - side_length // 2)
                bottom_right = (centroid[0] + side_length // 2,
                                centroid[1] + side_length // 2)
                # Update the image with the rectangle
                accumulated_squares_img = cv2.rectangle(
                    accumulated_squares_img, top_left, bottom_right,
                    color[::-1], thickness=-1)
                # 2- Define the color of the number
                font_color = choose_text_color(color[::-1])
                text = str(cluster_num)  # Number of the defect
                text_size = cv2.getTextSize(text, font, font_scale,
                                            font_thickness)[0]
                text_position = (
                    top_left[0] + (side_length - text_size[0]) // 2,
                    top_left[1] + (side_length + text_size[1]) // 2)
                # Update the image with the number
                accumulated_squares_img = cv2.putText(
                    accumulated_squares_img, text, text_position, font,
                    font_scale, font_color, font_thickness)
                # Convert the cv2 color to matplotlib for the legend
                matplotlib_color = (color[2]/255, color[1]/255,
                                    color[0]/255, 1.0)
                # 3- Update the legend
                legend_elements.append(
                    plt.Line2D([0], [0], marker='_',
                               color=matplotlib_color,
                               label=f'{cluster_num}: {round(perc)} %',
                               markerfacecolor=matplotlib_color,
                               markersize=5))

        # Get the image to plot
        mask_clusters = np.any(accumulated_squares_img != 0, axis=-1)
        black_defects_clusters_img = black_defects_img.copy()
        black_defects_clusters_img[mask_clusters] = (
            accumulated_squares_img[mask_clusters])
        # Show the original image
        if visualize_real_time:
            cv2.imshow(f"{caliper_folder_name}", original_reshaped_img)
        original_reshaped_img[mask_clusters] = (
            accumulated_squares_img[mask_clusters])

        # Prepare the image save
        path_to_save = os.path.join(
            caliper_folder_path,
            caliper_folder_name + f"_thresh_{ssim_threshold}" +
            f"_min_cluster_sz_{min_cluster_size}.png")
        if area_to_plot is not None:
            start_coord = area_to_plot[0]
            end_coord = area_to_plot[1]
            original_reshaped_img = original_reshaped_img[
                start_coord[1]:end_coord[1], start_coord[0]:end_coord[0]]

        # Plot the image
        plt.title(f"{images_per_caliper} images of {caliper_folder_name} \n"
                  f"SSIM Threshold = {ssim_threshold}/255, "
                  f"Min Cluster Size = {min_cluster_size}")
        plt.imshow(original_reshaped_img, cmap='gray')
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        ax.legend(handles=legend_elements,
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=False)
        plt.axis('off')
        plt.savefig(path_to_save, format='png', dpi=250)
        plt.show() if visualize_real_time else ""
        plt.close()
        cv2.waitKey() if visualize_real_time else ""

    def get_positives_and_negatives(
            self,
            min_cluster_size: int,
            clusters: dict[int, list, (int, int), str],
            GT_coords_filepath: str) -> Tuple[Set[Tuple[int, int]],
                                              Set[Tuple[int, int]],
                                              List[Tuple[str, str]]]:
        """
        Calculate true positives (TPs), false positives (FPs),
        and false negatives (FNs)
        considering predictions and ground truth areas.

        Parameters:
            big_clusters_centroids_num_samples
            (List[Tuple[Tuple[int, int], Tuple[int, int, int]]]):
            List of big clusters with centroids, sample counts,
            and color information.
            GT_coords_filepath (str): Path to the file containing ground truth
            coordinates manually inserted.

        Returns:
            Tuple[Set[Tuple[int, int]],
            Set[Tuple[int, int]], List[Tuple[str, str]]]: TPs, FPs, and FNs.
        """
        # Read and save the defective areas
        with open(GT_coords_filepath, 'r') as file:
            lines = file.readlines()

        if len(lines) > 0:  # If the caliper is defective
            # Initialize variables
            TPs = set()
            FPs = set()
            FNs = set()
            notFNs = set()
            # Initialize the list of all defective areas, each area is a tuple
            GT_areas = []  # [(x1, y1, x2, y2), (...)]
            for line in lines:
                # Save each area as a tuple of 4 coords
                coords_area = tuple(int(coord) for coord in line.split(','))
                # Append the area to all the other defective areas
                GT_areas.append(coords_area)

            if len(clusters) > 0:
                # Compare each GT defective area with predictions
                for centroid, (size, _, _, _) in clusters.items():
                    # Consider only the big clusters
                    if size >= min_cluster_size:
                        pred_x, pred_y = centroid
                        count_for_FP = 0
                        # Look for TP and FP
                        for GT_area in GT_areas:  # (x1, y1, x2, y2)
                            # Check if predictions are inside 'GT_area'
                            if (GT_area[0] <= pred_x <= GT_area[2]
                                    and GT_area[1] <= pred_y <= GT_area[3]):
                                TPs.add((pred_x, pred_y))
                                # Store 'GT_area' for later FNs computation
                                notFNs.add(GT_area)
                            else:  # Check if predictions are outside 'GT_area'
                                count_for_FP += 1
                        # A cluster is a FP if it does not fit in any 'GT_area'
                        if count_for_FP == len(GT_areas):
                            FPs.add((pred_x, pred_y))
                # FNs are all the lists in 'GT_areas' not in 'notFNs'
                if len(notFNs) > 0:
                    FNs = [(f"{GT_area[0]} -> {GT_area[2]}",
                            f"{GT_area[1]} -> {GT_area[3]}")
                           for GT_area in GT_areas
                           if GT_area not in notFNs]
            else:  # If no big clusters are found, all the GT_areas are FNs
                FNs = [(f"{GT_area[0]} -> {GT_area[2]}",
                        f"{GT_area[1]} -> {GT_area[3]}")
                       for GT_area in GT_areas]

        else:  # If the caliper is not defective
            TPs = set()  # Empty for sure
            FNs = set()  # Empty for sure
            FPs = set()  # To check
            # If there are centroids, they are all FPs
            if len(clusters) > 0:
                for centroid in clusters.keys():
                    FPs.add(centroid)

        return TPs, FPs, FNs

    def handle_single_acquisitions(
            self,
            original_reconstructed_dict: dict[int, tuple],
            caliper_folder: str,
            defects_visualization: dict[str, Any]) -> dict[tuple, int]:
        """
        """
        ssim_threshold = defects_visualization['ssim_threshold']
        min_cluster_size = defects_visualization['min_cluster_size']
        smoothing = defects_visualization['ssim_map']['smoothing']
        max_luminance = defects_visualization['ssim_map']['max_luminance']
        kernel_size = defects_visualization['ssim_map']['kernel_size']
        kernel_stdev = defects_visualization['ssim_map']['kernel_stdev']
        plot_flag = (defects_visualization['plot_defects_over_original_img'],
                     defects_visualization['plot_defects_over_ssim_map'])
        area_to_plot = defects_visualization['area_to_plot']
        dbscan_params = defects_visualization['dbscan_params']
        find_tp_fp_fn = defects_visualization['find_tp_fp_fn']
        store_insights_to_excel = \
            defects_visualization['store_insights_to_excel']
        exclude_reconstruction_errors = \
            defects_visualization['exclude_reconstruction_errors']
        max_distance_xy = defects_visualization['max_distance_xy']

        # Initialize the dictionary to accumulate the overall centroids
        overall_centroids = {}

        # Process each image of the caliper
        for number, photo_list in original_reconstructed_dict.items():
            # Find the original and reconstructed images in the list
            for bmp_file in photo_list:
                image_path = os.path.join(caliper_folder, bmp_file)
                bmp_filename = bmp_file[:-4]
                if "reconstructed" not in bmp_file:
                    original_frame = cv2.cvtColor(
                        cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
                elif "reconstructed" in bmp_file:
                    reconstructed_frame = cv2.cvtColor(
                        cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

            print(
                f"\n== Handling results on {bmp_filename}.bmp ... ==")

            t0_cpu = time.process_time()
            t0_real = time.perf_counter()
            ssim_map = self.compute_ssim(original_frame,
                                         reconstructed_frame,
                                         smoothing,
                                         max_luminance,
                                         kernel_size,
                                         kernel_stdev)
            print("CPU time for compute_ssim: "
                  f"{round(time.process_time()-t0_cpu, 3)} s")
            print("Real time for compute_ssim: "
                  f"{round(time.perf_counter()-t0_real, 3)} s")

            t0_cpu = time.process_time()
            t0_real = time.perf_counter()
            mask_defects_ssim_thresh, clusters_coords_ssim_thresh = (
                self.get_defects_mask(ssim_map, ssim_threshold))
            print("CPU time for get_defects_mask: "
                  f"{round(time.process_time()-t0_cpu, 3)} s")
            print("Real time for get_defects_mask: "
                  f"{round(time.perf_counter()-t0_real, 3)} s")

            # Store images with defects
            t0_cpu = time.process_time()
            t0_real = time.perf_counter()
            (original_red_defects, ssim_red_defects,
                black_defects_img, original_reshaped_img) = \
                self.draw_defective_pixels(original_frame,
                                           ssim_map,
                                           mask_defects_ssim_thresh)
            print("CPU time for draw_defective_pixels: "
                  f"{round(time.process_time()-t0_cpu, 3)} s")
            print("Real time for draw_defective_pixels: "
                  f"{round(time.perf_counter()-t0_real, 3)} s")

            # Plot images with red defects
            t0_cpu = time.process_time()
            t0_real = time.perf_counter()
            self.plot_defective_pixels(caliper_folder,
                                       bmp_filename,
                                       area_to_plot,
                                       ssim_threshold,
                                       original_red_defects,
                                       ssim_red_defects,
                                       plot_flag)
            print("CPU time for plot_defective_pixels: "
                  f"{round(time.process_time()-t0_cpu, 3)} s")
            print("Real time for plot_defective_pixels: "
                  f"{round(time.perf_counter()-t0_real, 3)} s")

            # Check if there are defects
            if np.any(mask_defects_ssim_thresh):
                t0_cpu = time.process_time()
                t0_real = time.perf_counter()
                # Store all the clusters
                clusters = self.get_defective_clusters(
                    dbscan_params, clusters_coords_ssim_thresh)
                # Verify that some clusters exist
                if len(clusters) != 0:
                    # Print how many big clusters are found
                    num_big_clusters = (
                        self.count_big_clusters(min_cluster_size, clusters))
                    print(f"\n{num_big_clusters} big clusters found")
                    # Print the shape of the big clusters
                    for centroid, (size, _, shape, _) in clusters.items():
                        if size >= min_cluster_size:  # Big cluster
                            print(f"Centroid {centroid}: ")
                            print(f"Width: {shape[0]} px")
                            print(f"Height: {shape[1]} px")
                            print(f"Width: {round(shape[0]*10/58, 3)} mm")
                            print(f"Heigth: {round(shape[1]*10/58, 3)} mm\n")
                    # Plot the big clusters
                    self.plot_big_clusters(
                        caliper_folder, bmp_filename, area_to_plot,
                        ssim_threshold, min_cluster_size,
                        black_defects_img, original_reshaped_img,
                        clusters, plot_flag)
                print("CPU time for computing and plotting clusters: "
                      f"{round(time.process_time()-t0_cpu, 3)} s")
                print("Real time for computing and plotting clusters: "
                      f"{round(time.perf_counter()-t0_real, 3)} s")
            else:
                clusters = {}
                num_big_clusters = 0
                print("No defects detected.")

            # Apply the overall defective centroids search
            overall_centroids = self.get_overall_centroids(
                min_cluster_size,
                exclude_reconstruction_errors,
                overall_centroids,
                clusters,
                max_distance_xy)

            def write_to_excel(clusters, TPs, FPs, FNs):
                """
                """
                # Build a dataframe for Excel
                excel_file_path = os.path.join(
                    self.weights_folder, 'defects_insights.xlsx')
                # Create the file if does not exist
                verify_excel_file_existence(excel_file_path)
                # Prepare data for Excel
                excel_clusters = {}
                for centroid, (size, _, _, flag) in clusters.items():
                    excel_clusters[centroid] = (size, flag)
                # Create the new sorted dictionary
                excel_clusters = dict(sorted(excel_clusters.items(),
                                             key=lambda item: item[1][0],
                                             reverse=True))
                data = {'Caliper Id': [caliper_folder.split("_")[-1]],
                        'Image Id': [number],
                        'SSIM Threshold': [ssim_threshold],
                        'Min Samples in Clusters': [min_cluster_size],
                        'Tot Clusters': [len(clusters)],
                        'Big Clusters': [num_big_clusters],
                        'TPs Count': [len(TPs)
                                      if type(TPs) is not str else ""],
                        'FPs Count': [len(FPs)
                                      if type(FPs) is not str else ""],
                        'FNs Count': [len(FNs)
                                      if type(FNs) is not str else ""],
                        'TPs': [TPs],
                        'FPs': [FPs],
                        'FNs': [FNs],
                        'All Clusters': [excel_clusters]}
                # Transform data into a dataframe
                df = pd.DataFrame(data)
                # Check if the sheet already exists
                if not verify_excel_sheet_existence(
                        file_path=excel_file_path,
                        sheet_name="calipers"):
                    # If the sheet does not exist, do not do df concatenation
                    df.to_excel(excel_file_path,
                                sheet_name="calipers",
                                index=False)
                else:
                    # Store the existing df
                    existing_df = pd.read_excel(excel_file_path,
                                                sheet_name="calipers")
                    # Concatenate the new df with the existing df
                    combined_df = pd.concat([existing_df, df],
                                            ignore_index=True)
                    # Add the new data to the existing one
                    combined_df.to_excel(excel_file_path,
                                         sheet_name="calipers",
                                         index=False)

            if find_tp_fp_fn:  # If the research is active
                coords_filepath = os.path.join(caliper_folder,
                                               'GT_coords.txt')
                TPs, FPs, FNs = self.get_positives_and_negatives(
                    min_cluster_size, clusters, coords_filepath)
            else:
                TPs = "The search is not active"
                FPs = "The search is not active"
                FNs = "The search is not active"

            if store_insights_to_excel:
                write_to_excel(clusters, TPs, FPs, FNs)

        # Transform numerical count into percentage over the total
        for centroid, count in overall_centroids.items():
            overall_centroids[centroid] = round(
                count/len(original_reconstructed_dict)*100, 1)

        # Sort the overall dict by the percentage of defects detection
        overall_centroids = dict(sorted(overall_centroids.items(),
                                        key=lambda x: x[1], reverse=True))

        return overall_centroids, black_defects_img, original_reshaped_img

    def handle_overall_visualization(self,
                                     defects_visualization: dict[str, Any]):
        """
        Handles the process of overall visualization.
        """
        # Save parameters of defects_visualization dictionary
        ssim_threshold = defects_visualization['ssim_threshold']
        min_cluster_size = defects_visualization['min_cluster_size']
        area_to_plot = defects_visualization['area_to_plot']
        min_perc_appearance = defects_visualization['min_perc_appearance']
        visualize_real_time = defects_visualization['visualize_real_time']

        def manage_weights_subfolders(subfolder_weights):
            """
            """
            # Create a dictionary for each image:
            # {file_number: [original_filename, reconstructed_filename]}
            original_reconstructed_dict = {}
            for filename in os.listdir(subfolder_weights):
                if filename.endswith(".bmp"):
                    # Extract the number from the file name
                    number = filename.split('_')[1]
                    # Add the file to the dictionary
                    # using the number as key
                    if number in original_reconstructed_dict:
                        original_reconstructed_dict[number].append(
                            filename)
                    else:
                        original_reconstructed_dict[number] = [filename]

            images_per_caliper = len(original_reconstructed_dict.items())

            return original_reconstructed_dict, images_per_caliper

        # Process each caliper subfolder
        for caliper_name in os.listdir(self.weights_folder):
            t0_cpu = time.process_time()
            t0_real = time.perf_counter()
            print("\n", caliper_name)
            caliper_folder = os.path.join(self.weights_folder, caliper_name)
            if (os.path.isdir(caliper_folder) and
                    not caliper_name.endswith(".ckpt")):
                # Get the couples of images and the number of acquisitions
                original_reconstructed_dict, images_per_caliper = (
                    manage_weights_subfolders(caliper_folder))

                # Handle each single bmp file before the overall results
                # and store the overall centroids to plot
                overall_centroids, black_defects_img, original_reshaped_img = \
                    self.handle_single_acquisitions(
                        original_reconstructed_dict,
                        caliper_folder,
                        defects_visualization)

                print(f"{caliper_name} - Clusters' detection percentage "
                      f"over {images_per_caliper} images:")
                # Print and store the detection percentage of the centroids
                centroids_filepath = os.path.join(
                    self.weights_folder,
                    caliper_name,
                    f'perc_clusters_{ssim_threshold}_{min_cluster_size}.txt')
                with open(centroids_filepath, 'w') as file:
                    file.write(str(caliper_name + "\n"))
                    for centroid, perc in overall_centroids.items():
                        print(f"{centroid}: {perc} %")
                        file.write(str(centroid) + ": " + str(perc) + " %\n")
                print("\n")

                print("CPU time for handling all the acquisitions: "
                      f"{round(time.process_time()-t0_cpu, 3)} s")
                print("Real time for handling all the acquisitions: "
                      f"{round(time.perf_counter()-t0_real, 3)} s")

                # Plot the overall defective centroids
                self.plot_overall_defective_clusters(caliper_folder,
                                                     caliper_name,
                                                     images_per_caliper,
                                                     area_to_plot,
                                                     ssim_threshold,
                                                     min_cluster_size,
                                                     black_defects_img,
                                                     original_reshaped_img,
                                                     overall_centroids,
                                                     min_perc_appearance,
                                                     visualize_real_time)


class BodyHandFeatures():
    """
    This class implements the production features for the body/hand detection
    model.
    """

    def __init__(self,
                 weights_folder: str,
                 camera: CameraAcquisition) -> None:
        super().__init__(weights_folder,
                         camera)

    def perform_inference(self,
                          frame: np.ndarray,):
        # TODO Implement the steps to perform inference on the current frame
        #  using the trained model
        print("Performing inference on current frame using trained model...")

    def handle_results(self,
                       results: Any):
        # TODO Implement the steps to handle the inference results as needed
        print(f"Handling inference results: {results}")

    def get_results(self,
                    preprocessing_setup: dict[str, Any] | None = None,
                    image_path: str | None = None):
        if self.camera:
            print("Getting current frame...")
            # Get current frame from camera
            frame = self.camera.frame
        else:
            print("Getting current frame from image...")
            # Get current frame from image
            frame = cv2.imread(image_path)
        # Preprocess current camera frame
        frame = FramePreprocessorOpenCV().preprocess_frame(
            frame=frame,
            preprocessing_setup=preprocessing_setup)
        # Perform inference on current camera frame
        self.perform_inference(frame)

    def start_streaming(self):
        # TODO Implement the steps to start streaming
        print("Starting streaming...")
        # Continuously display the camera frames with the
        # body and hand detection results using Mediapipe.


if __name__ == '__main__':

    # Retrieve the initial network architecture
    training_stage_cam1 = TrainingStage(
        build_stage=AutoencoderBuildStage(
            build_setup=None,
            training_setup=None,
            preprocessing_setup={
                'crop': (282, 416, 741, 1205),  # (y1, y2, x1, x2)
                'grayscale': True,
                'blur': (5, 5, 1),
                # resize[0] = crop[3] - crop[2]
                # resize[1] = crop[1] - crop[0]
                'resize': (464, 132),  # (1072, 98)
                # padding[0] = padding[1] = (resize[0] - resize[1]) / 2
                'padding': (166, 166, 0, 0),  # (487, 487, 0, 0)
            }),
        train_stage=None
    )
    training_stage_cam1.set_initial_network()
    initial_network = training_stage_cam1.initial_network

    weights_folder_cam1 = os.path.join(
        "app", "data", "weights", "cam_1")

    autoencoder_features = AutoencoderFeatures(
        weights_folder=weights_folder_cam1,
        initial_network=initial_network
    )

    autoencoder_features.handle_results()
