# KIAM_WEEK_07
KIAM_WEEK_07
# Telegram Data Pipeline with DBT and YOLO Object Detection

## Overview
This project implements a complete data pipeline for scraping, processing, transforming, and serving data collected from public Telegram channels related to Ethiopian medical businesses. The pipeline includes:

- **Data Scraping**: Extract messages and images from Telegram channels.
- **Data Cleaning and Transformation**: Process and standardize the data using DBT (Data Build Tool).
- **Object Detection**: Use YOLO for object detection in images collected from Telegram.
- **API Exposure**: Serve the collected and processed data via FastAPI.

## Tech Stack
- **Python** (Telethon, Pandas, OpenCV, FastAPI, SQLAlchemy, Torch)
- **DBT** (Data Transformation)
- **YOLOv5** (Object Detection)
- **SQLite/PostgreSQL** (Database Storage)

## Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install telethon pandas sqlalchemy opencv-python torch torchvision fastapi uvicorn dbt
```

### 2️⃣ Telegram Scraping Setup
1. Get your Telegram API credentials from [My Telegram](https://my.telegram.org/).
2. Update `api_id` and `api_hash` in `telegram_scraper.py`.
3. Run the scraper:
```bash
python telegram_scraper.py
```

### 3️⃣ DBT Setup for Data Transformation
1. Initialize a DBT project:
```bash
dbt init my_project
```
2. Configure the database connection in `profiles.yml`.
3. Define models in `models/`.
4. Run DBT transformations:
```bash
dbt run
dbt test
```

### 4️⃣ Object Detection with YOLO
1. Clone YOLO repository:
```bash
git clone https://github.com/ultralytics/yolov5.git && cd yolov5
pip install -r requirements.txt
```
2. Run detection:
```bash
python detect.py --source sample_image.jpg --weights yolov5s.pt
```

### 5️⃣ Expose API with FastAPI
1. Start FastAPI server:
```bash
uvicorn main:app --reload
```
2. Access API at `http://127.0.0.1:8000`

## Folder Structure
```
project/
├── telegram_scraper.py  # Scrapes Telegram channels
├── dbt_project/         # DBT transformation logic
├── yolov5/              # YOLO object detection
├── main.py              # FastAPI service
├── database.py          # Database setup
├── models.py            # DB models
└── README.md            # Project documentation
```

## Future Improvements
- Implement real-time streaming from Telegram.
- Deploy the FastAPI service using Docker.
- Extend object detection to classify detected objects.

---
This project provides a complete end-to-end data pipeline from data collection to transformation and visualization. 🚀

