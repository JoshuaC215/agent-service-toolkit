import json
import re
import unicodedata
from datetime import datetime
from typing import Literal

import requests
from json_repair import repair_json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from core import get_model, settings


class AgentState(MessagesState, total=False):
    messages: str | None = None


class MeetingRisks(BaseModel):
    description: str = Field(description="Description of the identified risk")


class MeetingFollowUps(BaseModel):
    description: str = Field(description="Description of the follow-up task")
    owner: str = Field(description="Person responsible for the follow-up task")
    due_date: str = Field(description="Due date for the follow-up task")


class MeetingAttendees(BaseModel):
    name: str = Field(description="Name of the attendee")


class MeetingDeadlines(BaseModel):
    description: str = Field(description="Description of the deadline")
    date: str = Field(description="Date by which the deadline must be met")


class MeetingQuestions(BaseModel):
    question: str = Field(description="The question raised during the meeting")
    raised_by: str = Field(description="Person who raised the question")
    status: str = Field(description="Current status of the question (e.g., Answered, Unanswered)")


class MeetingTasks(BaseModel):
    description: str = Field(description="Description of the task to be completed")
    assigned_to: str = Field(description="Person assigned to complete the task")
    priority: str = Field(description="Priority level of the task (e.g., High, Medium, Low)")


class Breakdownn(BaseModel):
    """Contents of the Meeting Summary"""

    tasks: list[MeetingTasks] = Field(
        description="List of all identified tasks with descriptions, assignees, and priorities"
    )
    decisions: list[MeetingRisks] = Field(
        description="List of all decisions made during the meeting"
    )
    questions: list[MeetingQuestions] = Field(
        description="List of all questions raised, including who raised them and their current status"
    )
    insights: list[MeetingRisks] = Field(
        description="List of all insights gained, explaining the rationale behind decisions"
    )
    deadlines: list[MeetingDeadlines] = Field(
        description="List of all deadlines with descriptions and due dates"
    )
    attendees: list[MeetingAttendees] = Field(
        description="List of all persons attending the meeting"
    )
    follow_ups: list[MeetingFollowUps] = Field(
        description="List of follow-up tasks with descriptions, owners, and due dates"
    )
    risks: list[MeetingRisks] = Field(description="List of all identified risks with descriptions")
    agenda: list[str] = Field(
        description="List of all top-level discussion points covered in the meeting"
    )
    meeting_name: str = Field(description="The official title of the meeting")
    description: str = Field(description="A short summary of the meeting's purpose and key topics")
    summary: str = Field(
        description="A detailed summary of the meeting, including major tasks, decisions, and outcomes"
    )


class MeetingSummary(BaseModel):
    """Summary of a meeting transcript"""

    Breakdown: Breakdownn = Field(description="The Contents of the Meeting summary")


current_date = datetime.now().strftime("%B %d, %Y")

json_example = """{"Breakdown": {"tasks": [{"description": "Prepare a report on the status of the procedural bylaw 2410.","assigned_to": "CEO","priority": "High"},{"description": "Follow up with the public works department regarding the tree asset projections.","assigned_to": "Nomar","priority": "Medium"},{"description": "Gather names for the two vacant positions on the Northeast Red Watershed District Committee.","assigned_to": "Council Members",    "priority": "Medium"},{    "description": "Draft the policy changes for the new community grant system.",    "assigned_to": "Grants Officer",    "priority": "High"},{    "description": "Schedule a meeting with local business leaders to discuss economic growth initiatives.",    "assigned_to": "Economic Development Manager","priority": "Low"}
            ],
            "decisions": [
                {
                    "description": "The agenda for the meeting was approved unanimously."
                },
                {
                    "description": "The council activity reports for September were received as information."
                },
                {
                    "description": "The additional cost of $33,300 for the Settlers Road Bridge Crossing project will be included in the 2025 capital budget."
                },
                {
                    "description": "Nonprofit organizations and community service groups will receive grants for 2024 as listed."
                },
                {
                    "description": "The municipal recreation center expansion was approved with an amended budget."
                },
                {
                    "description": "The council decided to allocate $15,000 for the local library digital resources upgrade."
                }
            ],
            "questions": [
                {
                    "question": "What is the status of the procedural bylaw 2410?",
                    "raised_by": "Janet",
                    "status": "Unanswered"
                },
                {
                    "question": "Why are the costs for the Settlers Road Bridge Crossing project increasing?",
                    "raised_by": "Andy",
                    "status": "Answered",
                    "answer": "The costs are increasing due to unforeseen costs and additional decommissioning requirements for the existing infrastructure."
                },
                {
                    "question": "What is the Society of Ivan Franco?",
                    "raised_by": "Mark",
                    "status": "Answered",
                    "answer": "It used to be an active community club located off Warren Hill Road, and discussions are ongoing about obtaining that land."
                },
                {
                    "question": "When will the public works department complete the tree asset projections?",
                    "raised_by": "Councilor Miller",
                    "status": "Pending"
                },
                {
                    "question": "How is the council planning to address the growing number of emergency motor vehicle collisions?",
                    "raised_by": "Glenn",
                    "status": "Unanswered"
                }
            ],
            "insights": [
                {
                    "description": "The council is committed to supporting seniors' activities and improving lodging for seniors in the community.",
                },
                {
                    "description": "There is a need for a daycare center in the RM, and land has been identified for this purpose.",
                },
                {
                    "description": "The increase in emergency motor vehicle collisions is concerning and needs further investigation.",
                },
                {
                    "description": "Public interest in developing additional recreational trails continues to grow.",
                },
                {
                    "description": "The community expressed concerns about rising utility rates in the region.",
                }
            ],
            "deadlines": [
                {
                    "description": "Submit names for the two vacant positions on the Northeast Red Watershed District Committee.",
                    "date": "2024-10-15"
                },
                {
                    "description": "Prepare the report on the procedural bylaw 2410 for the next meeting.",
                    "date": "2024-10-15"
                },
                {
                    "description": "Submit the draft policy changes for the new community grant system.",
                    "date": "2024-11-01"
                },
                {
                    "description": "Submit the budget proposal for the library digital resources upgrade.",
                    "date": "2024-10-20"
                }
            ],
            "attendees": [
                {
                    "name": "Mr. Mayor"
                },
                {
                    "name": "Councilor Miller"
                },
                {
                    "name": "Councilor Fuels"
                },
                {
                    "name": "Councilor Kaczynski"
                },
                {
                    "name": "Councilor Warren"
                },
                {
                    "name": "Councilor Lee"
                },
                {
                    "name": "Mark"
                },
                {
                    "name": "Melinda"
                },
                {
                    "name": "Andy"
                },
                {
                    "name": "Glenn"
                },
                {
                    "name": "Janet"
                },
                {
                    "name": "Nomar"
                },
                {
                    "name": "Public Works Director"
                },
                {
                    "name": "Grants Officer"
                }
            ],
            "follow_ups": [
                {
                    "description": "Follow up with the finance team regarding the budget approval for the Settlers Road Bridge Crossing project.",
                    "owner": "CEO",
                    "due_date": "2024-10-18"
                },
                {
                    "description": "Prepare a detailed report on the emergency motor vehicle collisions for the next meeting.",
                    "owner": "Public Works Director",
                    "due_date": "2024-10-15"
                },
                {
                    "description": "Meet with the daycare development committee to review the proposed land options.",
                    "owner": "Planning Department",
                    "due_date": "2024-10-22"
                },
                {
                    "description": "Organize a public forum on recreational trail development.",
                    "owner": "Community Engagement Coordinator",
                    "due_date": "2024-10-30"
                }
            ],
            "risks": [
                {
                    "description": "There is a risk of budget overruns due to unforeseen costs in ongoing projects."
                },
                {
                    "description": "Potential delays in the Settlers Road Bridge Crossing project could impact future budgets."
                },
                {
                    "description": "A shortage of qualified contractors may delay the municipal recreation center expansion."
                },
                {
                    "description": "Uncertainty around the future of federal infrastructure funding could affect long-term projects."
                }
            ],
            "agenda": [
                "Invocation and land acknowledgement",
                "Approval of the agenda",
                "Adoption of the minutes from the previous meeting",
                "Reports from council activities",
                "Departmental reports",
                "Question period",
                "Consent agenda",
                "Settlers Road Bridge Crossing project discussion",
                "Northeast Red Watershed District Committee appointments",
                "2024 nonprofit community grants discussion",
                "Public forum planning for recreational trail development",
                "2025 capital budget planning",
                "Utility rate increase concerns",
                "Closing of the meeting"
            ],
            "meeting_name": "October 2024 Municipal Council Meeting",
            "description": "This meeting covered several key topics including updates on ongoing infrastructure projects, community grant allocations, and recreational development initiatives. Key decisions were made regarding the Settlers Road Bridge Crossing, the municipal recreation center expansion, and the allocation of funds for the local library's digital resources. Questions and concerns about rising utility rates and the growing number of emergency motor vehicle collisions were raised. Several tasks, follow-ups, and deadlines were established to address ongoing issues, and risks were identified for future project planning.",
            "summary": "The meeting involved high-priority tasks such as preparing a procedural bylaw report, addressing public works follow-ups, and drafting policy changes for the community grant system. Key decisions included funding allocations for infrastructure projects and community services. Several questions were raised about project cost increases and emergency incidents, while the council focused on issues like recreational trail development and senior support. Insights indicated growing public concerns over utility rates, and various risks to project budgets and timelines were discussed."
            }
        }
    """

parser = PydanticOutputParser(pydantic_object=MeetingSummary)

instructions = f"""
    Based on the user transcription data you have access to, you need to produce a Breakdown of Categories:

    Tasks: Tasks with varying priorities, owners, and due dates.
    Example task assignments include preparing reports, setting up meetings, and submitting proposals.

    Decisions: Important decisions made during the meeting
    Decisions include vendor choice, marketing strategy, and budget approval.

    Questions: Questions raised during the meeting, with their status (answered/unanswered).
    Answered questions include additional context in the form of answers.

    Insights: Insights based on the conversation, ranging from sales performance to concerns about deadlines.
    Each insight refer back to the exact part of the conversation.

    Deadlines: Upcoming deadlines related to the budget, product launch, and client presentation.
    This helps track time-sensitive matters.

    Attendees: Attendees who attended the meeting
    This tracks attendance and their respective roles.

    Follow-ups: Follow-up tasks assigned to individuals after the meeting, each with a due date.
    Follow-up items focus on clarifying budget, design, and scheduling next actions.

    Risks: Risks identified during the meeting, each with potential impacts on the project.
    These include risks like budget overruns, delays, and potential staff turnover.

    Agenda: A list of the agenda items covered in the meeting.
    The agenda provides a structured overview of the topics discussed. You need to extract as many items as you can, some might have 1-2 items, and some might 10, so make sure to capture every point.

    Meeting Name: The title of the meeting, reflecting its official designation. This gives a clear identifier for the meeting, often including a specific date or purpose, such as "October 2024 Municipal Council Meeting."

    Description: A high-level overview of the meeting’s purpose and key areas of focus. The description captures the essential topics discussed, decisions made, and the overall scope of the meeting, such as infrastructure updates, budget approvals, and key community concerns.

    Summary: A brief consolidation of the main points and outcomes from the meeting. The summary encapsulates the flow of the meeting, including major tasks, decisions, and action points, along with any significant challenges or risks highlighted, offering a concise review of the meeting’s results.

    You must format your output as a JSON data, like:

    {json_example}

    Only return machine-readable json code, no newline characters or similar.

    The input message is the transcript of a meeting. Only answer contents in german. Today's date is {current_date}.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL), config["configurable"].get("api_key", None))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    tst = JSONCleaner()
    try:
        pass
        response.content = tst.clean_json(response.content)
    except Exception as e:
        print(e)

    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.set_entry_point("model")
agent.set_finish_point("model")

better_meeting_mind = agent.compile(checkpointer=MemorySaver())


class JSONCleaner:
    def clean_json(self, json_str):
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start == -1 or end == -1:
            msg = "Invalid JSON string: Missing '{' or '}'"
            raise ValueError(msg)
        try:
            json_str = json_str[start : end + 1]
            json_str = self._remove_control_characters(json_str)
            json_str = self._normalize_unicode(json_str)
            json_str = self._validate_json(json_str)

            return str(repair_json(json_str))
        except Exception as e:
            msg = f"Error cleaning JSON string: {e}"
            raise ValueError(msg) from e

    def _remove_control_characters(self, s: str) -> str:
        return re.sub(r"[\x00-\x1F\x7F]", "", s)

    def _normalize_unicode(self, s: str) -> str:
        return unicodedata.normalize("NFC", s)

    def _validate_json(self, s: str) -> str:
        try:
            json.loads(s)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON string: {e}"
            raise ValueError(msg) from e
        return s
