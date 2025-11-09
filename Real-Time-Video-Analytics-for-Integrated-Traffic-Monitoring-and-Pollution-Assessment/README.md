# Real-Time Video Analytics for Integrated Traffic Monitoring, Management, and Pollution Assessment
**Introduction:**
This repository contains the code and resources for a comprehensive project on real-time video analytics aimed at integrated traffic monitoring, management, and pollution assessment. The project addresses multiple problem statements related to AI-based automatic vehicle parking assistance, traffic monitoring across different live cameras, and measuring pollution based on vehicle count.

## **Problem Statements:**

1. **AI-Based Automatic Vehicle Parking Assistance:**

  - Develop an AI system to assist drivers in parking their vehicles automatically.
  - Utilize Object Detection and cameras to detect available parking spaces in real-time.
  - Provide users entering the parking area with real-time information on the number of available parking spaces.

2. **Monitoring the Traffic across different live cameras:**
  - Create a traffic monitoring system using Object Detection and live camera feeds.
  - Analyze and report real-time traffic conditions.
  - Identify traffic events for actionable insights to traffic management authorities and commuters.

3. **Measuring Pollution Based on Vehicle Count:**

  - Design a solution to estimate pollution levels based on the number and type of vehicles in a given area.
  - Analyze live traffic data and vehicle counts to predict pollution trends.
  - Provide real-time pollution estimates, alerts for high pollution levels, and data for environmental agencies to implement pollution control measures.

## **Prerequisites:**

- Colab for Model Training (Object Detector and Machine Learning)
- Visual Studio Code for productionized code development and CLI execution
- Python environment
- OpenCV for plotting bounding box coordinates and labeling
- Streamlit to deploy the Pipeline

## **Project Phases:**

1. **Dataset Collection:**
Collect dataset consisting of images of different vehicles using Roboflow and Kaggle.
2. **Data Preprocessing:**
Apply appropriate preprocessing steps such as data filtering and augmentation to enhance dataset quality.
3. **Model Building/Training:**
Train YOLOv8 model for Object Detection.
Develop Machine Learning models for pollution prediction based on vehicle counts.
4. **Model Evaluation:**
Evaluate model performance using metrics like accuracy, precision, recall, F1 score, mean average precision (mAP), and intersection over union (IoU).
5. **Model Conversion:**
Convert models to ONNX and OpenVINO formats for deployment.
6. **Deployment and Testing:**
Deploy models in Streamlit for real-time monitoring and assessment.
Test models on livestream and pre-recorded videos to validate performance and accuracy.
7. **Presentation and Reporting:**
Prepare reports and presentations detailing project overview, data collection, preprocessing, model building, evaluation, and deployment phases.

## **Code Structure:**

- **Parking.py:** Code for AI-based automatic vehicle parking assistance.
- **Vehicle_Detection.py:** Code for traffic monitoring across different live cameras.
- **Pollution_prediction.py:** Code for measuring pollution based on vehicle count.
- **Vehicle Detection Training:** Contains code for training YOLOv8 model for vehicle detection using Roboflow dataset.
- **Machine Learning DataSet Building:** Includes code for generating synthetic dataset and training linear regression model for pollution prediction.
- **Pollution_prediction.py:** Comprehended all the models used above into this leading to genearating web application

## **How to Run:**
1. **Upload Files/Folders to VS Code:**
   Upload all files and folders related to the project to your Visual Studio Code environment.
2. **Import Requirements:**
   Before running the code, make sure to import the requirements from the requirements.txt file to ensure all necessary dependencies are installed.
       pip install -r requirements.txt
3. **Run the Code:**
  Once the environment is set up, use the following command-line arguments to execute the desired problem statement.
    
        streamlit run Pollution_prediction.py

4. **Upload and Process:**
Once the code file is executed, a web application will open. Upload the relevant files, set your threshold value for accurate image detection, and click on "Process."

**Limitations:**
The vehicle detection model requires further training for improved accuracy. This model assumes all vehicles are petrol-fueled.

## **References:**

- **Environmental Protection Agency (EPA) NEI Data:**

    - [EPA NEI Data 2017](https://enviro.epa.gov/envirofacts/embed/nei?pType=SECTOR&pReport=county&pState=&pPollutant=&pPollutant=CO2&pPollutant=CH4&pPollutant=N2O&pPollutant=SF6&pSector=&pSector=12&pSector=14&pSector=28&pSector=54&pYear=2017&pCounty=&pTier=&pWho=NEI)

    - [2020 NEI Data](https://enviro.epa.gov/envirofacts/embed/nei?pType=SECTOR&pReport=county&pState=&pPollutant=&pPollutant=CO2&pPollutant=CH4&pPollutant=N2O&pPollutant=SF6&pSector=&pSector=12&pSector=14&pSector=28&pSector=54&pYear=2020&pCounty=&pTier=&pWho=NEI)

- **EPA Air Emissions Inventories:**

    - [2020 NEI Supporting Data and Summaries](https://www.epa.gov/air-emissions-inventories/2020-nei-supporting-data-and-summaries)

- **Federal Highway Administration (FHWA):**

    - [Highway Statistics 2022](https://www.fhwa.dot.gov/policyinformation/statistics/2022/rc1c.cfm)
    - [State Motor-Vehicle Registrations - 2022](https://www.fhwa.dot.gov/policyinformation/statistics/2022/mv1.cfm)


## **Conclusion:**
This project provides a comprehensive approach to real-time video analytics for traffic monitoring, management, and pollution assessment. By following the outlined phases and leveraging the provided resources, users can contribute to the advancement of automatic vehicle parking assistance technology and environmental sustainability efforts.
