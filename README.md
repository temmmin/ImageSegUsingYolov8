# Image Segmentation Using Yolov8 
This model is designed for steganography & DFC(Display Field Communication)


A project for image segmentation using YOLOv8. This repository demonstrates how to segment images efficiently with state-of-the-art deep learning models.

## Features
- YOLOv8-based image segmentation.
- This model is designed for steganography and DFC using images.
- This model is used for image region segmentation within a display.

## Conference Paper
**Title**: Image Segmentation Using YOLOv8 for Display-Field Communication (DFC) 
**Paper**: [Conference Paper Link](https://ieeexplore.ieee.org/document/10774043)

## References
This project utilizes the YOLOv8 model. For more details, visit the official YOLOv8 GitHub repository:

- [YOLO Official GitHub Repository](https://github.com/ultralytics/ultralytics)
- [YOLOv8 Official Hompage]([https://github.com/ultralytics/ultralytics](https://docs.ultralytics.com/ko/models/yolov8/))


## trained Weights

The trained weights for this project can be downloaded from the following link:

- [Download trained Weights using Custom dataset](https://drive.google.com/file/d/1MtDCr5guhAzoD9s4G-XIF445vmTzLTzg/view?usp=drive_link)
- train1.pt model was traiend with custom datasets

After downloading, place the weights file in the `weights/` directory to use them with the scripts.

## Usage
1. Install the dependencies listed in requirements.txt using "pip install -r requirements.txt"
2. Ensure that the paths in test_yolov8.py, such as `model_path`, `image_path`, and `save paths (save_final, save_cont,save_nask)`, are correctly set to match your file structure before running the script.
3. You can set warping size with `height` & `width` parameters
4.  Run test_yolov8.py


## Example
1. Original Image
![Original Image](/example/1.jpg)

2. Mask Image
![Mask Image](/example/mask.png)

3. Contour Result
![Contour Image](/example/cont.png)

4. Warping Result

![Warping Image](/example/result.png)



