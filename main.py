import cv2
from ultralytics import YOLO
import time
import pandas as pd
from collections import deque
import os
import matplotlib.pyplot as plt # New import for plotting
import numpy as np

def main():
    """
    Main function to detect vehicles, count them, and perform time-series prediction.
    """
    # --- Part 1: Real-time Data Collection ---
    # This part remains mostly the same as before.
    print("--- Part 1: Real-time Data Collection ---")
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully.")

    interested_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    video_path = 'traffic.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    traffic_data = deque(maxlen=1000)
    start_time = time.time()

    window_name = "YOLOv8 Traffic Analyzer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting video analysis. Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4, classes=interested_classes, verbose=False)
        detections = results[0].boxes
        vehicle_count = len(detections)
        elapsed_time = time.time() - start_time
        
        traffic_data.append({
            'timestamp': elapsed_time,
            'vehicle_count': vehicle_count,
            'frame_number': cap.get(cv2.CAP_PROP_POS_FRAMES)
        })

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, 
                    f"Vehicles: {vehicle_count}", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (0, 255, 255),
                    2, 
                    cv2.LINE_AA)

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

    df = pd.DataFrame(list(traffic_data))
    output_path = 'data/traffic_metrics.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Traffic metrics saved to {output_path}")

    # --- Part 2: Time-Series Prediction and Analysis ---
    print("\n--- Part 2: Time-Series Prediction and Analysis ---")

    # Check if the data file was created
    if not os.path.exists(output_path):
        print(f"Error: Data file {output_path} not found. Cannot proceed with analysis.")
        return

    # Load the data into a pandas DataFrame
    df = pd.read_csv(output_path)
    print("Data loaded for analysis.")

    # A simple way to represent a time-series is to resample the data.
    # Here, we'll group by time to get average vehicle counts per second.
    # Note: For this project, our video is short so this is a demonstration.
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df_resampled = df['vehicle_count'].resample('1S').mean().ffill() # 1-second resampling, forward-fill missing data

    # --- Simple Time-Series Prediction (Auto-Regressive Model) ---
    # We will predict the next 5 seconds of vehicle counts.
    # This is a very simplified AR model for demonstration.
    # A real AR model would be more complex, but this illustrates the concept.
    
    # We'll use the last 5 data points to predict the next 5.
    n_predict = 5
    historical_data = df_resampled.tail(n_predict).values
    
    predictions = []
    current_state = historical_data.copy()
    
    for _ in range(n_predict):
        # A simple AR model: the next value is a weighted sum of previous values.
        # This is a basic illustration; a real model would be trained to find these weights.
        next_pred = np.mean(np.array(current_state))
        predictions.append(next_pred)
        
        # Update the state to include the new prediction
        current_state = np.asarray(current_state)  # Ensure ndarray type
        current_state = np.append(current_state[1:], next_pred)
    
    # Create prediction timestamps
    last_timestamp = df_resampled.index[-1]
    prediction_timestamps = [last_timestamp + pd.Timedelta(seconds=i+1) for i in range(n_predict)]
    
    print("\nPrediction Results:")
    for ts, pred in zip(prediction_timestamps, predictions):
        print(f"Time: {ts.strftime('%H:%M:%S')}, Predicted Vehicles: {pred:.2f}")

    # --- Visualization ---
    # Plotting the data and predictions to visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled.values, label='Historical Data', color='blue')
    
    # Plotting predictions
    plt.plot(prediction_timestamps, predictions, label='Predictions', color='red', linestyle='--')
    
    plt.title('Traffic Flow: Historical vs. Predicted Vehicle Count')
    plt.xlabel('Time')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

