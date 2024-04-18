# Convolutional Autoencoder for Unsupervised Defect Detection on Brake Calipers

Artificial Intelligence, particularly Deep Learning models, has shown remarkable potential in revolutionizing industrial automation. This thesis focuses on leveraging such models, specifically Convolutional Autoencoders, for defect detection in industrial components. The research addresses the automation of quality control in brake calipers within a typical automotive production line setting.

The primary objective is to develop an automatic defect detection system capable of identifying various imperfections, such as deformities, bubbles, and scratches, on the brake calipers. A Convolutional Autoencoder is trained to decode input images of brake calipers, reconstructing them to remove potential defects. A difference function, accounting for contrast, brightness and structure, is then applied between the original and reconstructed images to identify candidate defective areas. Subsequently, a clustering algorithm categorizes these candidate areas as either defects or non-defects.

Experimental setup involves a custom-built workstation equipped with a camera, adjustable lighting and caliper fixtures for image acquisition. Data augmentation during training enhances the model’s ability to handle variability in environmental lighting and caliper positioning.

The developed methodology holds promise for streamlining quality control processes in industrial automation, contributing to reduced manual intervention and enhanced energy efficiency on production lines. However, challenges remain in adapting the system to diverse lighting conditions and caliper positions. Variability in external lighting and slight inconsistencies in caliper positioning pose significant hurdles, highlighting the need for further refinement to ensure robust performance across different operational scenarios.

