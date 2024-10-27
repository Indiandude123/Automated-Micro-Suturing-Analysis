# Automated Micro-Suturing Analysis

This project applies computer vision techniques to automate the analysis of micro-suturing, focusing on measuring precision and consistency across suturing attempts.

## Project Structure
- **data/**: Contains datasets used for training and validation.
- **scripts/**: Holds Python scripts for model training, data preprocessing, and analysis.
- **results/**: Stores output files such as analysis metrics, images, and visualizations.


## Requirements
Make sure you have task4.py in the same directory as this script and that it includes the necessary extract_image_features function. Install any required libraries (like OpenCV) if not already installed:

```bash
pip install opencv-python
```

## Running the Script
This script performs two tasks:

- Generate a CSV summarizing suture features for each image in a specified directory.
- Compare images based on their extracted features and output comparison results to a CSV.

```bash
python main.py <task> <input_path> <output_csv>
```

- <task>: Specify 1 or 2, where:
- 1 generates an output CSV summarizing features of all images in the specified directory.
- 2 compares pairs of images based on input from an existing CSV and writes results to an output CSV.
- <input_path>:
For task 1: The directory path containing images.
For task 2: Path to the input CSV file containing the image paths for comparison.
- <output_csv>: The name/path of the output CSV file.

