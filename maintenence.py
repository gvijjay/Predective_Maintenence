# anomaly_scheduler.py

import pandas as pd
from datetime import datetime, timedelta
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

# ---------- Timestamp Fix ----------
def convert_timestamp_format(input_path: str, output_path: str):
    try:
        df = pd.read_csv(input_path)

        # Auto-detect and parse mixed formats
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', dayfirst=True)

        # Save in ISO format
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        df.to_csv(output_path, index=False)
        print(f"✅ Timestamp format fixed and saved to {output_path}")
    except Exception as e:
        print(f"❌ Error while processing timestamp format: {e}")


# ---------- Load API Key ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ---------- Data Loaders ----------
class AnomalyDataTool:
    def __init__(self, path):
        convert_timestamp_format(path, path)  # Fix timestamps first
        self.df = pd.read_csv(path, parse_dates=["timestamp"])
        self.df.rename(columns={"timestamp": "anomaly_time", "Machine_name": "machine_id"}, inplace=True)
        self.df["estimated_repair_time_hr"] = 2

    def get_anomalies(self, date: str = None):
        df = self.df[self.df["Anomaly"] == "YES"].copy()
        if date:
            day = pd.to_datetime(date).date()
            df = df[df["anomaly_time"].dt.date == day]
        return df

    def has_anomalies(self, date: str = None):
        return not self.get_anomalies(date).empty

class TechnicianAvailabilityTool:
    def __init__(self, path):
        self.df = pd.read_csv(path, parse_dates=["availability_time"])
        self.df.rename(columns={"tasks_per_day": "max_tasks_per_day"}, inplace=True)
        self.df["tasks_assigned"] = 0

    def get_available_technicians(self, anomaly_time):
        candidates = self.df[
            (self.df["availability_time"] <= anomaly_time) &
            (self.df["tasks_assigned"] < self.df["max_tasks_per_day"])
        ]
        return candidates.sort_values("availability_time")

    def assign_task(self, tech_id, scheduled_end):
        idx = self.df[self.df["technician_id"] == tech_id].index[0]
        self.df.at[idx, "availability_time"] = scheduled_end
        self.df.at[idx, "tasks_assigned"] += 1

# ---------- Scheduling Logic ----------
class AssignmentTool:
    def __init__(self):
        self.schedule = []

    def assign(self, anomaly, technician):
        start_time = max(anomaly["anomaly_time"], technician["availability_time"])
        end_time = start_time + timedelta(hours=anomaly["estimated_repair_time_hr"])
        self.schedule.append({
            "machine_id": anomaly["machine_id"],
            "anomaly_time": anomaly["anomaly_time"],
            "technician_id": technician["technician_id"],
            "technician_name": technician["name"],
            "technician_email": technician["gmail"],
            "location": technician["location"],
            "scheduled_start": start_time,
            "scheduled_end": end_time
        })
        return end_time

    def get_schedule_df(self):
        return pd.DataFrame(self.schedule)

# ---------- Save Output ----------
class ScheduleFormatterTool:
    @staticmethod
    def save(schedule_df, path):
        schedule_df.to_csv(path, index=False)

# ---------- LangChain Tool Wrappers ----------
def schedule_anomalies(date_str: str = "") -> str:
    anomaly_tool = AnomalyDataTool("sensor_data_test_processed.csv")
    tech_tool = TechnicianAvailabilityTool("Tech_availability.csv")
    assign_tool = AssignmentTool()

    print(f"[DEBUG] Date input received: {date_str}")
    anomalies = anomaly_tool.get_anomalies(date_str)

    if anomalies.empty:
        return f"No anomalies found on {date_str}"

    for _, anomaly in anomalies.iterrows():
        techs = tech_tool.get_available_technicians(anomaly["anomaly_time"])
        if techs.empty:
            continue
        technician = techs.iloc[0]
        scheduled_end = assign_tool.assign(anomaly, technician)
        tech_tool.assign_task(technician["technician_id"], scheduled_end)

    df = assign_tool.get_schedule_df()
    ScheduleFormatterTool.save(df, "schedule.csv")
    return f"✅ Schedule saved to schedule.csv for date: {date_str}"

def check_anomaly_presence(date_str: str = "") -> str:
    anomaly_tool = AnomalyDataTool("sensor_data_test_processed.csv")
    print(f"[DEBUG] Checking anomalies for: {date_str}")
    has_anomaly = anomaly_tool.has_anomalies(date_str)
    return "Anomalies detected." if has_anomaly else f"No anomalies found on {date_str}."

# ---------- LangChain Agent ----------
def Scheduler_agent():
    tools = [
        Tool(
            name="ScheduleAnomalies",
            func=schedule_anomalies,
            description="Schedules technicians to anomalies based on optional date input and technician availability regardless of location"
        ),
        Tool(
            name="CheckAnomalyPresence",
            func=check_anomaly_presence,
            description="Checks whether there are any anomalies detected optionally on a specific date"
        )
    ]
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=api_key
    )
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

# ---------- Main ----------
if __name__ == "__main__":
    agent = Scheduler_agent()
    print("🔍 Running Anomaly Detection Agent...")
    result = agent.run("Schedule the persons for the anomaly detected machine for 1st March 2025 only.")
    print(f"Agent Response: {result}")
