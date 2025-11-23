🚦 AI-Powered Predictive Traffic Flow & Anomaly Detection
Project Overview
This is an end-to-end AI system designed to analyze and predict traffic flow from a video source. By combining real-time computer vision with time-series forecasting and anomaly detection, this project demonstrates a modern approach to urban mobility challenges.

The system processes video from a simulated traffic camera, extracts key metrics like vehicle count and density, and uses this data to predict future traffic conditions and identify unusual events that could lead to congestion or accidents.

Key Features:

Real-time Vehicle Detection: Uses a pre-trained YOLOv8 model to accurately detect cars, buses, motorcycles, and trucks in a video stream.

Time-Series Data Generation: Transforms real-time detection data into a structured time-series dataset.

Predictive Analytics: Implements a simple time-series model to forecast future vehicle counts based on historical trends.

Unsupervised Anomaly Detection: Utilizes the Isolation Forest algorithm to automatically identify unexpected spikes or drops in traffic flow, which could indicate accidents or blockages.

Data Visualization: Generates a plot to visually represent historical traffic data, predictions, and detected anomalies.

Technologies & Libraries Used
Python 3.10: The primary development language.

OpenCV (opencv-python): For video processing and displaying the output.

YOLOv8 (ultralytics): A state-of-the-art object detection model for vehicle classification and counting.

Pandas (pandas): The core library for handling and analyzing time-series data.

Scikit-learn (scikit-learn): Used to implement the Isolation Forest model for anomaly detection.

Matplotlib (matplotlib): For data visualization and plotting the results.

NumPy (numpy): For numerical operations, especially with time-series data.

Project Structure
traffic_flow_analyzer/
├── .venv/                      # Python Virtual Environment
├── main.py                     # Main application script
├── requirements.txt            # Lists all Python dependencies
├── README.md                   # This file
├── models/                     # For pre-trained models
│   └── yolov8n.pt
└── data/                       # For storing generated data
    └── traffic_metrics.csv

How to Run the Project
Prerequisites
Python 3.10 installed on your system.

A video file of traffic (e.g., traffic.mp4) in the project root directory.

Setup
Create and Activate a Virtual Environment:

py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

Create Folders: Ensure the data and models folders exist in your project directory. You can create them with:

mkdir data models

Create requirements.txt: Create this file in the project root with the following content:

opencv-python
ultralytics
pandas
scikit-learn
numpy

Install Dependencies:

pip install -r requirements.txt

Execution
Place your traffic.mp4 video file in the traffic_flow_analyzer directory.

Run the main script from your activated virtual environment:

python main.py

The script will:

Process the video, displaying bounding boxes and a vehicle count.

After the video ends, it will save the metrics to data/traffic_metrics.csv.

Load the data, perform a time-series analysis, and detect anomalies.

Display a graph of the results with predictions and anomalies highlighted.

Project Walkthrough
The project is divided into two parts:

Part 1: Real-time Data Collection
The main.py script first opens the video file and enters a processing loop. In each frame, it uses the YOLOv8 model to detect vehicles. It then counts these vehicles and appends the count, along with a timestamp, to a pandas DataFrame. This effectively converts an unstructured video stream into a structured, numerical dataset.

Part 2: Predictive & Anomaly Analysis
Once the video processing is complete, the script loads the generated traffic_metrics.csv file. It then performs two key analyses:

Time-Series Prediction: A simple Auto-Regressive (AR) model is used to forecast the vehicle count for the next few seconds, demonstrating the system's predictive capability.

Anomaly Detection: An Isolation Forest model is fit on the historical data to identify outliers (anomalous data points) that deviate from the normal traffic flow pattern.

The final output is a graph that visually represents the entire process, making the results of both the prediction and anomaly detection clear and intuitive.

Future Enhancements
Live Webcam Integration: Adapt the code to process a live stream from a webcam or IP camera.

Advanced Forecasting Models: Replace the simple AR model with more sophisticated time-series models like LSTM or Prophet for more accurate long-term predictions.

Pothole/Hazard Detection: Integrate an additional object detection model to specifically identify potholes and other road anomalies, building on the initial traffic analysis.

Dynamic Thresholds: Implement an adaptive system where anomaly detection thresholds change based on the time of day or known traffic patterns.

Geospatial Mapping: Integrate a GPS data stream to tag detected anomalies with precise location, enabling real-world mapping and reporting.
