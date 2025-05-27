import streamlit as st
import pandas as pd
from io import BytesIO
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Agent imports
from agents import AnomolyDetection_agent
from maintenence import Scheduler_agent
from alert import Alert_agent

load_dotenv()

st.set_page_config(page_title="📡 Chat Scheduler App", layout="wide")
st.title("🤖 Chat-Driven Maintenance Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# File upload
uploaded_file = st.file_uploader("Upload sensor data file (CSV)", type="csv")
if uploaded_file:
    file_path = "sensor_data_test.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded and saved as sensor_data_test.csv. You can now ask to detect anomalies.")

# Define tool functions with chat history integration
def run_anomaly_detection(prompt: str) -> str:
    """Run anomaly detection and store result in chat history"""
    st.session_state.messages.append({"role": "assistant", "content": "Running anomaly detection..."})
    
    agent = AnomolyDetection_agent()
    result = agent.run("Run anomaly detection on sensor_data_test.csv using sensor_anomaly_detector tool.")
    
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

def run_scheduler(prompt: str) -> str:
    """Run scheduler and store result in chat history"""
    st.session_state.messages.append({"role": "assistant", "content": "Scheduling technicians..."})
    
    agent = Scheduler_agent()
    result = agent.run(f"Schedule technicians for {prompt} based on the schedule anomalies tool.")
    
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

def run_alerts(prompt: str) -> str:
    """Run alert system and store result in chat history"""
    st.session_state.messages.append({"role": "assistant", "content": "Sending alerts..."})
    
    agent = Alert_agent()
    result = agent.run("send email alerts to technicians")
    
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

# Initialize tools
tools = [
    Tool(
        name="AnomalyDetection", 
        func=run_anomaly_detection, 
        description="Useful for when you need to detect anomalies in sensor data. Input should be a request to analyze sensor data."
    ),
    Tool(
        name="Scheduler", 
        func=run_scheduler, 
        description="Useful for when you need to schedule technicians for maintenance. Input should be details about the scheduling requirements."
    ),
    Tool(
        name="AlertAgent", 
        func=run_alerts, 
        description="send the alert email notifications to the technician."
    )
]

# Initialize agent with memory
agent = initialize_agent(
    tools=tools,
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    ),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.agent_memory,
    verbose=True
)

# Prompt input
user_input = st.chat_input("Ask about anomaly detection, scheduling, or sending alerts")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Determine if we should use a specific tool directly based on keywords
    use_direct_tool = None
    tool_response = None
    
    if "anomaly" in user_input.lower() or "detect" in user_input.lower():
        use_direct_tool = "AnomalyDetection"
    elif "schedule" in user_input.lower() or "technician" in user_input.lower():
        use_direct_tool = "Scheduler"
    elif "alert" in user_input.lower() or "email" in user_input.lower():
        use_direct_tool = "AlertAgent"
    
    if use_direct_tool:
        # Use the specific tool directly
        for tool in tools:
            if tool.name == use_direct_tool:
                tool_response = tool.func(user_input)
                break
    else:
        # Let the agent decide which tool to use based on conversation history
        tool_response = agent.run(user_input)
    
    # # If we didn't already add the response via the tool functions
    # if not any(msg["role"] == "assistant" and msg["content"] == tool_response for msg in st.session_state.messages):
    #     st.session_state.messages.append({"role": "assistant", "content": tool_response})


# Chat display
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            
            # Show relevant dataframes and download buttons with unique keys
            if "anomaly" in msg["content"].lower() and os.path.exists("sensor_data_test_processed.csv"):
                df = pd.read_csv("sensor_data_test_processed.csv")
                st.dataframe(df)
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                st.download_button(
                    "Download Anomaly CSV", 
                    buffer.getvalue(), 
                    file_name="sensor_data_test_processed.csv",
                    key=f"anomaly_download_{i}"  # Unique key for each button
                )
            
            if "schedule" in msg["content"].lower() and os.path.exists("schedule.csv"):
                df = pd.read_csv("schedule.csv")
                st.dataframe(df)
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                st.download_button(
                    "Download Schedule CSV", 
                    buffer.getvalue(), 
                    file_name="schedule.csv", 
                    key=f"schedule_download_{i}"  # Unique key for each button
                )