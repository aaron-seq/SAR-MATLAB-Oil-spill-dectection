# Oil Spill Detection and Segmentation in SAR Images

This project provides a MATLAB-based environment for detecting and segmenting oil spills in Synthetic Aperture Radar (SAR) images. It implements and compares various image processing techniques, allowing users to analyze SAR images, apply different algorithms, and evaluate their performance against ground truth data.

![Evaluation Metrics](https://user-images.githubusercontent.com/96207365/185741928-d8a9379d-d6f7-490b-8694-4d1b32435daf.jpg)

## About The Project

- The project consists of designing and implementing image processing techniques for the **detection** and **segmentation** of oil spills on **SAR** images.
- To verify the **correctness** and **precision** of these methods, the segmented images are compared to the **ground truth** masks from the dataset used in this [article](https://www.researchgate.net/publication/334715725_Oil_Spill_Identification_from_Satellite_Images_Using_Deep_Neural_Networks).
- The produced masks identify three main classes: **black** for the sea surface, **green** for land, and **cyan** for oil spills.
- A key feature is the ability for the user to **interactively fine-tune parameters** for each algorithm to improve segmentation results based on the specific characteristics of the image being analyzed.

![Overlapped Mask](https://user-images.githubusercontent.com/96207365/185742093-b89e98e3-aa9e-43fa-9b67-394317a99cc7.jpg)

---

## What is Synthetic Aperture Radar (SAR)?

Synthetic Aperture Radar (SAR) is a remote sensing technology used to create high-resolution images of the Earth's surface using radio waves. Unlike optical sensors, SAR is not dependent on sunlight and can operate day and night, as it actively emits microwave pulses and measures their reflections.

In the context of oil spill detection, SAR is an indispensable tool. When an oil spill occurs, it dampens the small capillary waves on the water's surface. This change makes the slick area smoother than the surrounding sea, causing it to reflect less radar signal back to the satellite (lower backscatter). This difference in backscatter makes the oil spill appear as a dark spot on the SAR image, allowing for its detection and delineation.

SAR's ability to operate regardless of weather (it can see through clouds) and lighting conditions makes it particularly valuable for continuous monitoring and rapid response during oil spill emergencies.

---

## Getting Started

### Prerequisites

- **MATLAB** (R2020a or newer is recommended)
- **Image Processing Toolbox™**

### Setup & Usage

1.  **Download the Dataset**: Make sure you have the oil spill dataset. The expected directory structure is:
    ```
    <your_dataset_directory>/
    └── train/
        ├── images/
        ├── images_with_land/
        ├── labels/
        └── labels_with_land/
    ```
2.  **Launch the Program**:
    - Open MATLAB.
    - Run the `main.m` script.
    - When prompted, select the top-level `<your_dataset_directory>` that you downloaded.
3.  **Follow the Menus**: Use the command window menus to:
    - Select the type of image to analyze (with or without land).
    - Choose a specific image by its number.
    - Select a segmentation algorithm.
    - After the initial result, you will be prompted to fine-tune the parameters to improve the segmentation.

---

## Segmentation Methods Implemented

The project explores two main scenarios based on the image type:

#### 1. Images with Only Sea and Oil Spills
- **Thresholding Segmentation**
    - *Manual Thresholding*: User manually selects pixels to define the threshold.
    - *Automatic Thresholding*: Thresholds are determined automatically from the image histogram.
    - *Local Adaptive Thresholding*: The threshold is varied across the image based on local statistics.
- **Superpixel Approach**: The image is divided into small, perceptually uniform regions (superpixels), which are then classified.
- **Fuzzy Logic Approach**: Uses fuzzy inference rules based on image gradients to detect edges.
- **K-means Clustering**: Pixels are grouped into a set number of clusters based on intensity, with the darkest cluster identified as the oil spill.

#### 2. Images with Land, Sea, and Oil Spills
- **Automatic Thresholding**: A combination of thresholding to identify both land and potential oil spill regions.
- **K-means Clustering**: Used to separate the image into multiple clusters, identifying land (brightest), sea, and oil spills (darkest).

![K-Means Clusters](https://user-images.githubusercontent.com/96207365/183259085-7a9be686-348b-4d3c-b775-c34ce0133a80.jpg)

---

## Results and Evaluation

Every segmentation is evaluated both qualitatively and quantitatively:

-   **Qualitative Evaluation**: The final segmented mask is overlaid on the original image and shown alongside the ground truth for visual comparison.
-   **Quantitative Evaluation**: Three standard metrics are calculated to measure the similarity between the computed mask and the ground truth:
    -   **Jaccard Index**
    -   **Sørensen-Dice Similarity**
    -   **BF (Boundary F1) Score**

The evaluation metrics are displayed with a visual comparison where:
-   **Green**: Represents the ground truth area.
-   **White**: Correctly identified segmentation (True Positive).
-   **Violet**: Incorrectly identified segmentation (False Positive).

---

## Documentation

For a detailed technical explanation of the original implementation, please see the [Code explanation (technical implementation).pdf](https://github.com/aaronseq12/MatlabOilspilldetection/blob/main/Code%20explanation%20(technical%20implementation).pdf) file. *Note: Some code structures may have changed as part of the repository improvement process.*

## Support

For any support, error corrections, etc., please email me at aaronsequeira12@gmail.com.
