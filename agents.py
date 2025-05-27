from langchain.tools import BaseTool
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from langchain.tools import BaseTool
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

THRESHOLD = 0.02
MODEL_PATH = "models/deep_autoencoder_anomaly.h5"

class SensorAnomalyTool(BaseTool):
    name: str = "sensor_anomaly_detector"
    description: str = "Detects anomalies in sensor data using a trained autoencoder model"

    def _run(self, input_file: str) -> str:
        df = pd.read_csv(input_file)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        scaler = MinMaxScaler()
        features = df[["vibration_amplitude_g", "frequency_hz", "humidity_percent", "temperature_celsius"]]
        scaled_data = scaler.fit_transform(features)
        reconstructions = model.predict(scaled_data)
        mse = np.mean(np.square(scaled_data - reconstructions), axis=1)

        df["MSE"] = mse
        df["Anomaly"] = np.where(mse > THRESHOLD, "YES", "NO")
        df["Reason"] = np.where(
            df["Anomaly"] == "YES",
            "Anomaly detected - Vibration: " + df["vibration_amplitude_g"].round(1).astype(str) +
            "g, Temp: " + df["temperature_celsius"].round(1).astype(str) + "C, Freq: " + df["frequency_hz"].round(1).astype(str) + "Hz",
            "Normal operation"
        )
        output_path = input_file.replace(".csv", "_processed.csv")
        df.to_csv(output_path, index=False)
        return output_path

    def _arun(self, input_file: str):
        raise NotImplementedError("Async not supported.")



def AnomolyDetection_agent():
    tools = [SensorAnomalyTool()]
    agent = initialize_agent(
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=api_key
        ),
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


import sys
def main():
    input_file = sys.argv[1]  # e.g., "sensor_data.csv"
    agent = AnomolyDetection_agent()

    print("🔍 Running Anomaly Detection Agent...")
    processed_file = agent.run(f"Run anomaly detection on {input_file} using sensor_anomaly_detector tool.")
    print(f"✅ Anomaly Detection Done: {processed_file}")


if __name__ == "__main__":
    main()
