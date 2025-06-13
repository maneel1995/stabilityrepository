# Eyeglasses Detection Pipeline

A distributed image processing pipeline that detects faces and identifies people wearing eyeglasses in images. 

## Overview

This project implements an end-to-end pipeline for processing images to detect people wearing eyeglasses. It uses:

- **YOLO** for face detection
- **CLIP** for eyeglass identification
- **Ray** for distributed processing

## Project Structure

- `src/data.py`: Downloads dataset files from Huggingface
- `src/model.py`: Contains the `ImageClassifier` class that processes images
- `src/inference.py`: Implements distributed offline batch infererence using the Ray Framework
- `src/load.py`: Uploads processed data to Huggingface

## Installation

```
git clone <repository-url>
cd stabilityrepository
pip install -r requirements.txt
```

## Getting Started

### Downloading data
This project uses the Wikimedia WIT dataset from Huggingface:

```bash
python src/ddata.py --full False --workers 4
```
Options:
- `--full`: Downloads the full dataset (330 files) if True, or just 2 files
- `--workers`: Number of parallel download threads (default:4)

The download process uses ThreadPoolExecutor for parallel downloads. 

### 2. Running Inference

Process the downloaded parquet files with the pipeline:

```bash
python src/inference.py --concurrency 3 --batch_size 16
```

Options:
- `--concurrency`: Number of concurrent inference workers (default: 1)
- `--batch_size`: Batch size for inference (default: 5)

This will process all images, detect faces, classify whether they contain people wearing eyeglasses and save the result to parquet files. Using the Ray framework means that we can scale this process to billions of images with data in disk and in a distributed cluster. 

### 3. Upload Results to Huggingface 

```bash
python src/load.py --repo_id your-username/image_predictions --output_dir data/output
```

Options:
- `--repo_id`: Target Huggingface dataset repo name
- `--output_dir`: Path to the output directory with parquet files (default: data/output/)
- `--public`: Make dataset public (default is private)

### Model Architecture

The pipeline uses two models:

1. **Yolov8 Face Detection**: Detects faces in images with bounding boxes
    - Pre-trained model from Huggingface(arnabdhar/YOLOv8-Face-Detection)
    - Used for its speed and accuracy in detecting human faces

2. **CLIP**: CLassifies whethere detected faces are wearing eyeglasses. Extremely flexible 
    - Pre-trained model from OpenAI(openai/clip-vit-base-patch32)
    - Leverages zero-shot classification capabilities through carefully crafted text prompts

### Architecture Choices

This project uses Yolov8 to detect if the image contains a person. If yes, then filter once again using a CLIP model. Then filter to check if the person is indeed wearing eyeglasses. This results in 91 images belonging to people with eyeglasses in the first two partitions and they can be found here. 

### Architecture Tradeoffs
While this pipeline uses a simple filtering process. Other simpler models are in MTCCN library. We could also have used HaarCascade for eyeglass detection. But, the accuracy was not high. Higher accuracy at the cost of detection of people with eyeglasses can be achieved by increasing the confidence of the CLIP model. Also, more prompt tuning can be done in the CLIP model. Other techniques  Running OpenCV HaarCascade or LBP classifiers in parallel: Easy to implement but their performance on real-world noisy images is poor, and may generate too many false detections.


