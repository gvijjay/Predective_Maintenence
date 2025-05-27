# Predective_Maintenence Chat application

A full-stack AI application integrating data preprocessing, deep learning (LSTM), LangChain-based agents with Google Generative AI, email notifications, and a Streamlit UI for real-time interaction and insights.

## 🎯 Goal:

Enable users to upload sensor data and interact via natural language to:

1.Detect anomalies

2.Schedule technicians

3.Notify via email



## 🔧 Features
These are the features that we have implemented in our project:


- 🤖 LangChain agents powered by Google Generative AI

- 📈 Time-series anomaly detection with LSTM Autoencoders

- 📤 Email alerts for anomalies

- 🔁 Conversational memory for contextual 

## Tech Stack
 The below libraries are the tech stack that we have used:
 
- Python, Pandas, NumPy, Scikit-learn

- TensorFlow / Keras (LSTM autoencoders)

- LangChain, Google GenAI API

- Streamlit for frontend UI

- SMTP for email alerts

- dotenv for environment management

## 🚀 Installation
The below are the steps that you have to follow to install the project.

#### * Clone the repo
`git clone https://github.com/gvijjay/Predective_Maintenence.git`

`cd <your-project-folder>`

#### * Create virtual environment
`python -m venv venv`

`source venv/bin/activate`  

##### on Windows: `venv\Scripts\activate`

#### * Install dependencies
`pip install -r requirements.txt`

## ⚙️ Environment Setup
Create a .env file in the root directory with the following keys:

`GOOGLE_API_KEY=your_google_api_key`

`EMAIL_USER=you@example.com`

`EMAIL_PASSWORD=your_password`

## ▶️ Running the App
You have to run the below command in the terminal for successfully running the application

`streamlit run app.py`


## 🧠 Agents Overview
### 1. Anomaly Detection Agent (AnomolyDetection_agent)

Input: `Sensor CSV (sensor_data_test.csv)`

Output: `sensor_data_test_processed.csv`

Logic:

  - Uses a trained deep autoencoder model to detect anomalies

  - Flags data points with high reconstruction error

  - Adds "Anomaly" and "Reason" columns to the dataset


### 2. Scheduling Agent (Scheduler_agent)

Input: `Anomaly CSV + Technician availability`

Output: `schedule.csv`

 Logic:

  - Reads all anomalies from the processed file

  - Assigns available technicians based on:

  - available_from ≤ anomaly time

  - Not exceeding max_tasks_per_day

  - Outputs scheduled technician details per machine


### 3. Alert Agent (Alert_agent)
   
Input: `schedule.csv + technician contact details`

Action: `Sends personalized email alerts to each technician`

Logic:

  - First-come-first-serve

  - Only notifies technicians scheduled for the earliest day

  - Skips if no technician or duplicate



## 🖥️ Streamlit Integration

- Uses `st.chat_input()` for natural prompt entry

- Agents are conditionally triggered based on keywords:

    "anomaly" → AnomalyDetection

    "schedule", "availability" → Scheduler

    "email", "alert" → Alert

- Memory is enabled `(ConversationBufferMemory)` to maintain chat history

## 📜 License

 MIT
