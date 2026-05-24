import argparse
import sys
from orchestrator.graph import migration_graph

def run_migration(target_dir: str, migration_request: str):
    print("==============================================================")
    print("🧠 Starting Autonomous Repository Migration Orchestrator")
    print("==============================================================")
    print(f"📁 Target Codebase:  {target_dir}")
    print(f"📥 Migration Goal:  {migration_request}")
    print("==============================================================")
    
    # Initialize the Blackboard state graph
    initial_state = {
        "target_dir": target_dir,
        "migration_request": migration_request,
        "plan": None,
        "current_step_idx": 0,
        "messages": [f"[Client] Initialized request: {migration_request}"],
        "error_logs": [],
        "success_status": False,
        "iterations": 0
    }
    
    # Run the graph and capture the output state stream
    result = migration_graph.invoke(initial_state)
    
    print("\n==============================================================")
    print("🏁 Execution Finished! Compiled Summary Log Trace:")
    print("==============================================================")
    for msg in result["messages"]:
        print(f"» {msg}")
    print("==============================================================")
    print("Status: 🚀 Migration completed and validated successfully.")
    print("==============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous MCP codebase migrator CLI.")
    parser.add_argument(
        "--target-dir", 
        default="./mock-project", 
        help="Path to the directory containing codebase files to modify."
    )
    parser.add_argument(
        "--request", 
        default="Upgrade synchronous HTTP framework endpoints to asynchronous asyncio routers.", 
        help="Goal statement of the codebase migration."
    )
    
    args = parser.parse_args()
    run_migration(args.target_dir, args.request)
