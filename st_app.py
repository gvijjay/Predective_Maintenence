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
from whatsapp_alert import Alert_agent_whatsapp
from data_scout import DataScout_agent, DataScout_agent_with_pdf

load_dotenv()

st.set_page_config(page_title="📡 Chat Scheduler App", layout="wide")
st.title("🤖 Chat-Driven Maintenance Agent")

# Sidebar with default flow example
with st.sidebar:
    st.header("🧭 Default Workflow Example")
    st.markdown("""
    1. **Upload sensor CSV** 📂
    2. **Enter prompt**: `Detect anomalies in uploaded file`
    3. **Enter prompt**: `Schedule technicians for detected anomalies`
    4. **Select methods (Email/WhatsApp)** 📬📲
    5. **Enter prompt**: `Send alert messages`
    """)

    st.divider()
    st.subheader("📊 Sample Data Creation Prompts")
    st.markdown("""
    - Excel:
      - `Generate 100 employee records with Name, Department, Salary, Joining Date`
      - `Create a product inventory list with columns: Product ID, Name, Category, Price, Stock`
    
    - PDF:
      - `Create a 2-page project report with Introduction, Methodology, Results, and Conclusion`
      - `Generate a summary document for AI trends in 2025`
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "notify_methods" not in st.session_state:
    st.session_state.notify_methods = []

# File upload
uploaded_file = st.file_uploader("Upload sensor data file (CSV)", type="csv")
if uploaded_file:
    file_path = "sensor_data_test.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded and saved as sensor_data_test.csv. You can now ask to detect anomalies.")

# Define tool functions with chat history integration
def run_anomaly_detection(prompt: str) -> str:
    st.session_state.messages.append({"role": "assistant", "content": "Running anomaly detection..."})
    agent = AnomolyDetection_agent()
    result = agent.run("Run anomaly detection on sensor_data_test.csv using sensor_anomaly_detector tool.")
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

def run_scheduler(prompt: str) -> str:
    st.session_state.messages.append({"role": "assistant", "content": "Scheduling technicians..."})
    agent = Scheduler_agent()
    result = agent.run(f"Schedule technicians for {prompt} based on the schedule anomalies tool.")
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

def run_alerts(prompt: str) -> str:
    result = ""
    options = st.session_state.get("notify_methods", [])
    if "Email" in options:
        st.session_state.messages.append({"role": "assistant", "content": "Sending email alert..."})
        email_agent = Alert_agent()
        result += email_agent.run("send email alerts to technicians") + "\n"
    if "WhatsApp" in options:
        st.session_state.messages.append({"role": "assistant", "content": "Sending WhatsApp alert..."})
        wa_agent = Alert_agent_whatsapp()
        result += wa_agent.run("send whatsapp alert to technician")
    st.session_state.messages.append({"role": "assistant", "content": result})
    return result

# Initialize tools
tools = [
    Tool(name="AnomalyDetection", func=run_anomaly_detection, description="Detect anomalies in sensor data."),
    Tool(name="Scheduler", func=run_scheduler, description="Schedule technicians for maintenance."),
    Tool(name="AlertAgent", func=run_alerts, description="Send alert notifications to technicians via email or WhatsApp.")
]

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

# Tabs
tab1, tab2 = st.tabs(["Main", "Data Creation Assistant"])

# Tab 1 - Original functionality
with tab1:
    st.session_state.notify_methods = st.multiselect("Select notification methods for alerts:", ["Email", "WhatsApp"], key="notify_multiselect")

    user_input = st.chat_input("Ask about anomaly detection, scheduling, or sending alerts")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        use_direct_tool = None
        tool_response = None

        if "anomaly" in user_input.lower() or "detect" in user_input.lower():
            use_direct_tool = "AnomalyDetection"
        elif "schedule" in user_input.lower() or "technician" in user_input.lower():
            use_direct_tool = "Scheduler"
        elif "alert" in user_input.lower() or "email" in user_input.lower() or "whatsapp" in user_input.lower():
            use_direct_tool = "AlertAgent"

        if use_direct_tool:
            for tool in tools:
                if tool.name == use_direct_tool:
                    tool_response = tool.func(user_input)
                    break
        else:
            tool_response = agent.run(user_input)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "anomaly" in msg["content"].lower() and os.path.exists("sensor_data_test_processed.csv"):
                df = pd.read_csv("sensor_data_test_processed.csv")
                st.dataframe(df)
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                st.download_button("Download Anomaly CSV", buffer.getvalue(), file_name="sensor_data_test_processed.csv", key=f"anomaly_download_{i}")
            if "schedule" in msg["content"].lower() and os.path.exists("schedule.csv"):
                df = pd.read_csv("schedule.csv")
                st.dataframe(df)
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                st.download_button("Download Schedule CSV", buffer.getvalue(), file_name="schedule.csv", key=f"schedule_download_{i}")

# Tab 2 - Data Creation Assistant
with tab2:
    st.title("Data Creation Assistant")
    st.markdown("Create Excel spreadsheets or PDF reports with AI")

    data_type = st.radio(
        "Select data type to create:",
        ("Excel", "PDF"),
        horizontal=True,
        key="data_creation_type"
    )

    if st.session_state.data_creation_type == "Excel":
        prompt = st.text_area(
            "Describe the data you want to generate (include field names and number of rows):",
            placeholder="E.g. Generate 25 patient records with fields: Name, Age, Gender, Diagnosis, Admission Date"
        )
    else:
        prompt = st.text_area(
            "Describe the PDF document you want to generate (include sections and content requirements):",
            placeholder="E.g. Create a 3-page medical report about diabetes with sections: Introduction, Symptoms, Treatment Options"
        )

    if st.button("Generate Data"):
        if prompt.strip() == "":
            st.warning("Please enter a description of what you want to generate")
        else:
            with st.spinner(f"Generating {st.session_state.data_creation_type} file..."):
                try:
                    if st.session_state.data_creation_type == "Excel":
                        agent = DataScout_agent()
                        response = agent.invoke(prompt)
                        if isinstance(response, dict) and 'output' in response:
                            file_path = response['output']
                            st.session_state.generated_file = {
                                'type': 'excel',
                                'path': file_path,
                                'name': os.path.basename(file_path)
                            }
                        else:
                            st.error("Failed to generate Excel file")
                    else:
                        agent = DataScout_agent_with_pdf()
                        response = agent.invoke(prompt)
                        if isinstance(response, dict) and 'output' in response:
                            file_path = response['output']
                            st.session_state.generated_file = {
                                'type': 'pdf',
                                'path': file_path,
                                'name': os.path.basename(file_path)
                            }
                        else:
                            st.error("Failed to generate PDF file")
                except Exception as e:
                    st.error(f"Error generating file: {str(e)}")

    if st.session_state.get("generated_file"):
        st.success(f"{st.session_state.data_creation_type} file generated successfully!")

        if st.session_state.generated_file['type'] == 'excel':
            df = pd.read_excel(st.session_state.generated_file['path'])
            st.dataframe(df.head())
            with open(st.session_state.generated_file['path'], "rb") as f:
                bytes_data = f.read()
            st.download_button("Download Excel File", data=bytes_data, file_name=st.session_state.generated_file['name'], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            with open(st.session_state.generated_file['path'], "rb") as f:
                bytes_data = f.read()
            st.download_button("Download PDF File", data=bytes_data, file_name=st.session_state.generated_file['name'], mime="application/pdf")

        if st.button("Clear Generated File"):
            st.session_state.generated_file = None
            st.rerun()
