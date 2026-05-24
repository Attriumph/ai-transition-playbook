from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from orchestrator.agents import SpecializedAgents, MigrationPlan

# Define the shared state dictionary (The Blackboard Pattern)
class MigrationState(TypedDict):
    target_dir: str
    migration_request: str
    plan: MigrationPlan
    current_step_idx: int
    messages: Annotated[List[str], operator.add]
    error_logs: List[str]
    success_status: bool
    iterations: int

# Define the Agent Node actions
def generate_plan_node(state: MigrationState) -> Dict[str, Any]:
    print(f"🌟 [Planner] Analyzing migration request for target directory: {state['target_dir']}")
    # Mocking LLM planner output structure for demonstration
    # In a real environment, this invokes the ChatOpenAI/ChatAnthropic model structured output
    mock_plan = MigrationPlan(
        goal="Migrate standard HTTP routes to async endpoint routers.",
        steps=[
            {
                "step_number": 1,
                "description": "Convert synchronous client connection to async standard.",
                "affected_files": ["client.py"],
                "is_completed": False
            },
            {
                "step_number": 2,
                "description": "Upgrade routes in app.py to async definitions.",
                "affected_files": ["app.py"],
                "is_completed": False
            }
        ]
    )
    return {
        "plan": mock_plan,
        "current_step_idx": 0,
        "messages": ["[Planner] Scaffolding plan: Converting routes to async."],
        "iterations": 0
    }

def execute_step_node(state: MigrationState) -> Dict[str, Any]:
    plan = state["plan"]
    idx = state["current_step_idx"]
    current_step = plan.steps[idx]
    
    print(f"🛠️ [Coder] Executing Step {current_step.step_number}/{len(plan.steps)}: {current_step.description}")
    print(f"👉 Target files: {current_step.affected_files}")
    
    # Simulating file reading and file writing via custom MCP server tools
    # If error logs exist, Coder self-corrects using historical traceback context
    if state["error_logs"]:
        print(f"⚠️ [Coder] Addressing error traceback: {state['error_logs'][-1]}")
        log_msg = f"[Coder] Applying patch self-correction for step {current_step.step_number}."
    else:
        log_msg = f"[Coder] Successfully wrote modifications to {', '.join(current_step.affected_files)}."
        
    return {
        "messages": [log_msg],
        "iterations": state["iterations"] + 1
    }

def validate_step_node(state: MigrationState) -> Dict[str, Any]:
    idx = state["current_step_idx"]
    plan = state["plan"]
    current_step = plan.steps[idx]
    
    print(f"🧪 [Tester] Validating Step {current_step.step_number} changes...")
    
    # Self-Correction Loop simulation:
    # First iteration of step 2 will fail to demonstrate compile/retry cycles.
    is_step_two = (current_step.step_number == 2)
    has_errors = is_step_two and (len(state["error_logs"]) == 0)
    
    if has_errors:
        error_msg = "SyntaxError: 'async' def routes cannot contain synchronous block return."
        print(f"❌ [Tester] Validation FAILED with compile traceback: {error_msg}")
        return {
            "error_logs": [error_msg],
            "messages": [f"[Tester] Step {current_step.step_number} validation failed. Dispatching traceback."]
        }
    else:
        print("✅ [Tester] Validation PASSED. Changes compile and unit tests successfully passed.")
        return {
            "error_logs": [],
            "messages": [f"[Tester] Step {current_step.step_number} validated successfully."],
            "current_step_idx": idx + 1
        }

# Conditional Edges Routing Decisions
def route_validation_results(state: MigrationState) -> str:
    # If the tester caught an error, route back to the coder for self-correction
    if state["error_logs"]:
        if state["iterations"] >= 5: # Guard to prevent infinite echo chamber loops
            print("🚨 [Orchestrator] Max iterations exceeded! Stopping execution graph.")
            return "abort"
        return "re-edit"
    
    # If no errors and we have processed all steps, complete the graph
    if state["current_step_idx"] >= len(state["plan"].steps):
        return "finalize"
    
    # Otherwise, move to the next planned step
    return "next-step"

# Construct the Graph Topology
workflow = StateGraph(MigrationState)

# Add Nodes
workflow.add_node("planner", generate_plan_node)
workflow.add_node("coder", execute_step_node)
workflow.add_node("tester", validate_step_node)

# Set Entry Point
workflow.set_entry_point("planner")

# Standard Edges
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "tester")

# Conditional Edges from Tester Validation
workflow.add_conditional_edges(
    "tester",
    route_validation_results,
    {
        "re-edit": "coder",      # Loop back to Coder to resolve issues
        "next-step": "coder",    # Move to the next migration step
        "finalize": END,         # Graceful complete exit
        "abort": END             # Emergency break stop
    }
)

# Compile the execution graph
migration_graph = workflow.compile()
