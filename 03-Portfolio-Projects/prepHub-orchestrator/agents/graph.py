from typing import List, Dict, Any, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Define STAR feedback structure
class STARMetric(BaseModel):
    situation_and_task: int = Field(description="Score from 1 to 5 evaluating context description.")
    action: int = Field(description="Score from 1 to 5 evaluating detail of actions executed.")
    result: int = Field(description="Score from 1 to 5 evaluating impact, metrics and results shared.")
    communication_rating: int = Field(description="Score from 1 to 5 on speech flow and confidence.")

class AssessmentReport(BaseModel):
    overall_score: float = Field(description="Average score of behavioral performance.")
    strengths: List[str] = Field(description="Key strengths logged during interview.")
    weaknesses: List[str] = Field(description="Key behavioral gaps or missed STAR opportunities.")
    improved_rewrite: str = Field(description="A model rewrite of a STAR answer that the candidate gave.")

# Custom Thread Reducer for merging live transcripts without corrupting metrics
def merge_dialogue_transcript(
    current_chat: List[Dict[str, str]], 
    new_turns: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Appends new conversational dialogue turns into the state memory pipeline.
    """
    return current_chat + new_turns

class InterviewState(TypedDict):
    job_target: str
    resume_context: str
    chat_history: Annotated[List[Dict[str, str]], merge_dialogue_transcript]
    current_question: str
    critic_notes: List[str]
    report: AssessmentReport
    turns_count: int

# --- Graph Nodes ---

def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """
    The Interviewer Agent. Analyzes the chat history and generates a realistic coding
    or behavioral follow-up question. Under the hood, it has access to web search tools.
    """
    print(f"👔 [Interviewer Agent] Formulating next step for: {state['job_target']}")
    
    # In production, this runs a prompt passing resume_context, job_target and chat_history
    # We simulate a dynamic, structured follow-up question here.
    next_question = "In your resume, you mention scaling a real-time event pipeline to 50k QPS. Can you walk me through your system architecture, specifically how you handled database connection pooling during spike loads?"
    
    return {
        "current_question": next_question,
        "chat_history": [{"role": "assistant", "content": next_question}],
        "turns_count": state["turns_count"] + 1
    }

def critic_node(state: InterviewState) -> Dict[str, Any]:
    """
    The Critic Agent. Silently reviews the user's latest response in the chat history,
    evaluating it against the STAR method metrics.
    """
    last_user_message = state["chat_history"][-1]["content"] if state["chat_history"] else ""
    print(f"📝 [Critic Agent] Auditing candidate's response: '{last_user_message[:40]}...'")
    
    # Analyze STAR gaps:
    notes = []
    if "result" not in last_user_message.lower() and "%" not in last_user_message:
        notes.append("Candidate failed to provide quantifiable metrics in the Result section of their answer.")
    else:
        notes.append("Good mention of quantifiable metrics.")
        
    return {
        "critic_notes": notes
    }

def coach_node(state: InterviewState) -> Dict[str, Any]:
    """
    The Coach Agent. Runs at the end of the session, aggregating all critic notes
    and dialogue history to output a structured STAR Assessment Report.
    """
    print("🎓 [Coach Agent] Compiling final performance assessment report...")
    
    report = AssessmentReport(
        overall_score=4.2,
        strengths=["Quantifiable scale figures quoted.", "Strong architecture breakdowns."],
        weaknesses=["Omitted technical trade-offs between RabbitMQ and Kafka."],
        improved_rewrite="I scaled our event pipeline to 50k QPS by introducing a Kafka event buffer. Rather than writing to Postgres directly, we batched writes using a Redis queue, reducing database locking times by 40%."
    )
    return {
        "report": report
    }

# --- Graph Flow Control ---

def route_interview_turn(state: InterviewState) -> str:
    # Set the interview length boundary (e.g., 4 dialogue turns)
    if state["turns_count"] >= 4:
        return "conclude"
    return "continue"

# --- Building the Graph ---

workflow = StateGraph(InterviewState)

# Define Nodes
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("coach", coach_node)

# Set Entrance
workflow.set_entry_point("interviewer")

# Define Flows
workflow.add_edge("interviewer", "critic")

workflow.add_conditional_edges(
    "critic",
    route_interview_turn,
    {
        "continue": "interviewer", # Loop back to ask the next question
        "conclude": "coach"        # Transition to compile the report
    }
)

workflow.add_edge("coach", END)

# Compile the Graph
interview_graph = workflow.compile()
