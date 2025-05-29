# whatsapp_notifier.py

import os
import pandas as pd
from dotenv import load_dotenv
from typing import Optional
from twilio.rest import Client
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

load_dotenv()

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SENT_LOG_FILE_WA = "notified_techs_wa.log"

class WhatsAppAlertTool:
    def __init__(self, schedule_path: str, tech_path: str):
        self.schedule_df = pd.read_csv(schedule_path, parse_dates=["scheduled_start", "scheduled_end"])
        self.technicians_df = pd.read_csv(tech_path)
        self.client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        self.notified_techs = self._load_notified_techs()

    def _load_notified_techs(self):
        if not os.path.exists(SENT_LOG_FILE_WA):
            return set()
        with open(SENT_LOG_FILE_WA, "r") as file:
            return set(line.strip() for line in file.readlines())

    def _mark_as_notified(self, tech_id):
        with open(SENT_LOG_FILE_WA, "a") as file:
            file.write(f"{tech_id}\n")

    def send_whatsapp(self, to_number: str, body: str):
        self.client.messages.create(
            body=body,
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            to=f"whatsapp:{to_number}"
        )

    def notify_one_whatsapp(self) -> str:
        if self.schedule_df.empty:
            return "⚠️ No schedule entries found."

        self.schedule_df = self.schedule_df.sort_values("scheduled_start")

        for _, row in self.schedule_df.iterrows():
            tech_id = str(row['technician_id'])
            if tech_id in self.notified_techs:
                continue

            tech_info = self.technicians_df[self.technicians_df['technician_id'] == int(tech_id)]
            if tech_info.empty:
                return f"❌ Technician with ID {tech_id} not found."

            whatsapp = tech_info.iloc[0]['mobile_no']
            name = tech_info.iloc[0]['name']

            maintenance_date = row['scheduled_start'].strftime('%Y-%m-%d')
            message = (
                f"Hello {name},\n\n"
                f"You have been scheduled to attend the anomaly on Machine ID: {row['machine_id']} on {row['scheduled_start']}\n"
                f"Scheduled Start: {row['scheduled_start']}\n"
                f"Scheduled End: {row['scheduled_end']}\n\n"
                "Please make the necessary arrangements.\n\nRegards,\nScheduler Bot"
            )

            self.send_whatsapp(whatsapp, message)
            self._mark_as_notified(tech_id)
            return f"✅ WhatsApp notification sent to {name} ({whatsapp}) for machine {row['machine_id']}."

        return "🔁 No new technicians to notify via WhatsApp."

def whatsapp_alert(_: str = "") -> str:
    alert_tool = WhatsAppAlertTool("schedule.csv", "Tech_availability.csv")
    return alert_tool.notify_one_whatsapp()

def Alert_agent_whatsapp():
    tools = [
        Tool(
            name="SendWhatsAppScheduleAlerts",
            func=whatsapp_alert,
            description="Sends one WhatsApp notification to the first unnotified technician in schedule.csv"
        )
    ]
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=GEMINI_API_KEY
    )
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

if __name__ == "__main__":
    agent = Alert_agent_whatsapp()
    print("📲 Sending WhatsApp Alert...")
    result = agent.run("send whatsapp alert to technician")
    print(result)
