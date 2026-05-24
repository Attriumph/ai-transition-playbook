import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Using LangChain & Pydantic primitives to define structured outputs and prompt interfaces.
class MigrationStep(BaseModel):
    step_number: int = Field(description="Sequential number of the migration step.")
    description: str = Field(description="Detailed explanation of what needs to be changed.")
    affected_files: List[str] = Field(description="List of files that will be read or modified in this step.")
    is_completed: bool = Field(default=False, description="Tracking completion state.")

class MigrationPlan(BaseModel):
    goal: str = Field(description="The primary objective of the code migration.")
    steps: List[MigrationStep] = Field(description="Step-by-step sequential migration plan.")

# Specialized System Instructions
PLANNER_SYSTEM_PROMPT = """
You are the Lead AI Migration Architect. Your task is to analyze a codebase migration request and generate a structured, sequential plan.
You must:
1. Identify dependencies: determine which modules need to be updated first.
2. Break down the task into small, isolated steps where each step affects 1-3 files maximum.
3. Keep focus on backward compatibility and preventing runtime regressions.
"""

CODER_SYSTEM_PROMPT = """
You are an expert Refactoring and Coding Agent. Your task is to read the codebase using MCP tools and perform precise, AST-aware refactoring.
When editing:
1. Preserving all existing code logic, interfaces, docstrings, and formatting unless explicitly instructed.
2. Make targeted edits to the exact lines requiring upgrades.
3. Ensure all new functions have proper type annotations and docstrings.
4. Output your edits using the 'write_file' or target patch tool.
"""

TESTER_SYSTEM_PROMPT = """
You are the QA and Compiler Validation Agent. Your job is to verify that code migrations are structurally sound.
You will:
1. Examine compiler outputs, syntax trees, or run test suites.
2. If compilation fails or a test crashes, capture the complete traceback.
3. Classify the error (e.g., Import Error, Syntax Error, Type Mismatch) and write clear instructions for the Coder agent to help it self-correct.
"""

class SpecializedAgents:
    @staticmethod
    def get_planner_prompt() -> str:
        return PLANNER_SYSTEM_PROMPT
        
    @staticmethod
    def get_coder_prompt() -> str:
        return CODER_SYSTEM_PROMPT
        
    @staticmethod
    def get_tester_prompt() -> str:
        return TESTER_SYSTEM_PROMPT
