import pandas as pd
import numpy as np
import pickle
import streamlit as st
from ultralytics import YOLO
import cv2

st.title('Real-Time Traffic Monitoring and Pollution Assessment')
st.subheader('AI-Based Automatic Vehicle Traffic Monitoring and Pollution Prediction')

# Sidebar
st.sidebar.title('User Inputs')

# File uploader
video_file = st.sidebar.file_uploader('Upload a video file', type=['mp4', 'avi'])

# Select box for model selection
model_option = st.sidebar.selectbox(
    'Choose a Model',
    ('Vehicle Detection', 'Pollution Prediction')
)
# Additional select box for pollution prediction
# Slider for setting confidence threshold
confidence_threshold = st.sidebar.slider(
    'Confidence Threshold', 0.0, 1.0, 0.5
)

flag1 = 0
flag2 = 0
flag3 = 0
flag4 = 0
total = 0
total_flag = 0
emission_prediction = 0

# Load Models Based on User Selection
@st.cache_resource
def load_model(model_option):
    if model_option == 'Vehicle Detection':
        return YOLO('weights/vehicle_detection_best.pt')
    elif model_option == 'Pollution Prediction':
        return YOLO('weights/vehicle_detection_best.pt')  # Load the same YOLO model for vehicle detection

yolo_model = load_model(model_option)

# Placeholder for displaying video frame
video_placeholder = st.empty()

# Placeholder for displaying vehicle count
counter_placeholder = st.empty()

def process_frame(frame, model, confidence_threshold):
    results = model.predict(frame, save=False, conf=confidence_threshold, imgsz=640)
    coordinates = []
    if len(results[0]) >= 1:
        a = results[0].boxes.data.cpu().detach().numpy()
        px = pd.DataFrame(a).astype("float")
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            cls = int(row[5])
            coordinates.append([x1, y1, x2, y2, cls])
    return coordinates

def preprocess_count_data(counts, ml_model):
    with open('weights\\scaler.pkl', "rb") as f:
        scaler = pickle.load(f)
    #scaler = joblib.load('weights\\scaler.pkl')
    counts_df = pd.DataFrame(counts, columns=['count_of_car', 'count_of_truck', 'count_of_bike', 'count_of_bus'])
    counts_scaled = scaler.transform(counts_df)
    prediction = ml_model.predict(counts_scaled)
    prediction = round(prediction[0], 2)
    return prediction

# Button to process video
if st.sidebar.button('Process Video'):

    st.write(f'Processing video with {model_option} model...')
    st.write(f'Confidence Threshold: {confidence_threshold}')
    
    # Example processing function
    def process_video(video_path, model_option, confidence_threshold, model_option2):
        global flag1, flag2, flag3, flag4, total_flag, emission_prediction

        cap = cv2.VideoCapture(video_path)
        
        total_bike_count = 0
        total_car_count = 0
        total_transport_vehicles = 0
        total_bus_count = 0

        flag_bike = 0
        flag_car = 0
        flag_transport = 0
        flag_bus = 0  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame based on the selected model_option
            coordinates = process_frame(frame, model_option, confidence_threshold)

            #results = yolo_model(frame)
            
            Bike_count = 0
            Car_count = 0
            Bus_count = 0
            transport_vehicles = 0

            for coord in coordinates:
                x1, y1, x2, y2, cls = coord
                color_mapping = {
                    0: (0, 255, 0),
                    1: (0, 0, 255),
                    3: (255, 255, 0)
                }
                color = color_mapping.get(cls, (160, 32, 240))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if cls == 1:
                    cv2.putText(frame, "Bike", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    Bike_count += 1
                elif cls == 0:
                    cv2.putText(frame, "Car", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 100, 255), 2)
                    Car_count += 1
                elif cls == 2:
                    cv2.putText(frame, "Transport_Vehicle", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 100, 255), 2)
                    transport_vehicles += 1
                elif cls == 3:
                    cv2.putText(frame, "Bus", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 100, 255), 2)
                    Bus_count += 1

            if flag_bike < Bike_count:
                total_bike_count += Bike_count - flag_bike
            flag_bike = Bike_count

            if flag_car < Car_count:
                total_car_count += Car_count - flag_car
            flag_car = Car_count

            if flag_transport < transport_vehicles:
                total_transport_vehicles += transport_vehicles - flag_transport
            flag_transport = transport_vehicles

            if flag_bus < Bus_count:
                total_bus_count += Bus_count - flag_bus
            flag_bus = Bus_count
                
            ml_model_path = 'weights/LR_model.pkl'
    
            with open(ml_model_path, "rb") as f:
                ml_model = pickle.load(f)
            counts = np.array([[total_car_count, total_transport_vehicles, total_bike_count, total_bus_count]])
            emission_prediction = preprocess_count_data(counts, ml_model)

            # Display the frame in the placeholder
            video_placeholder.image(frame, channels='BGR')
            if model_option2 == 'Pollution Prediction':
                counter_placeholder.text(f'Total Bike count: {total_bike_count}\nTotal Car count: {total_car_count}\nTotal Transport Vehicle count: {total_transport_vehicles}\nTotal Bus count: {total_bus_count}\nVehicle Count: {total_car_count}\nCO2 emitted (g/km): {emission_prediction}')
            elif model_option2 == 'Vehicle Detection':
                counter_placeholder.text(f'Total Bike count: {total_bike_count}\nTotal Car count: {total_car_count}\nTotal Transport Vehicle count: {total_transport_vehicles}\nTotal Bus count: {total_bus_count}\nVehicle Count: {total_car_count}')
    

            if total_flag == 0:
                flag_variables = [flag1, flag2, flag3, flag4]
            else:
                flag_variables = [flag1, flag2, flag3, flag4]
            counts = [Bike_count, Car_count, transport_vehicles, Bus_count]

            for i in range(4):
                if flag_variables[i] < counts[i]:
                    total_flag = total_flag + counts[i] - flag_variables[i]
                    flag_variables[i] = counts[i]
                else:
                    flag_variables[i] = counts[i]

            flag1, flag2, flag3, flag4 = flag_variables

        cap.release()
    
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        process_video(video_path, yolo_model, confidence_threshold, model_option)