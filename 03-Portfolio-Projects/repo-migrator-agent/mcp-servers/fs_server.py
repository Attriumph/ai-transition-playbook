import os
import re
import ast
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP Server
mcp = FastMCP("AST-Aware Filesystem Server")

@mcp.tool()
def read_file_segment(path: str, start_line: int, end_line: int) -> str:
    """
    Reads a specific line segment of a file. Optimized for large codebases to avoid context window bloating.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Ensure indices are within bounds
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)
        
        segment = "".join(lines[start_line - 1:end_line])
        return f"--- File Segment: {path} (Lines {start_line}-{end_line}) ---\n{segment}"
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Writes content directly to a file, creating any missing parent directories.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to '{path}'."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"

@mcp.tool()
def search_regex(dir_path: str, pattern: str, file_extension: str = ".py") -> str:
    """
    Performs a high-speed regex search across all files matching the specified extension in a directory.
    """
    if not os.path.exists(dir_path):
        return f"Error: Directory '{dir_path}' does not exist."
    
    results = []
    try:
        regex = re.compile(pattern)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(file_extension):
                    full_path = os.path.join(root, file)
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for idx, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{full_path}:{idx} -> {line.strip()}")
        
        if not results:
            return f"No matches found for pattern '{pattern}' in directory '{dir_path}'."
        return "\n".join(results[:100]) # Cap at 100 matches
    except Exception as e:
        return f"Error searching directory '{dir_path}': {str(e)}"

@mcp.tool()
def parse_python_ast(path: str) -> str:
    """
    Parses a Python file using the Abstract Syntax Tree (AST) module. 
    Exposes high-level syntax nodes (classes, function names, import namespaces) without dumping raw code content.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    if not path.endswith(".py"):
        return "Error: AST parser only supports Python files (.py)."
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            node = ast.parse(f.read(), filename=path)
        
        structure = []
        for child in node.body:
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
                structure.append(f"Import: {', '.join(names)}")
            elif isinstance(child, ast.ImportFrom):
                names = [alias.name for alias in child.names]
                structure.append(f"ImportFrom: from {child.module} import {', '.join(names)}")
            elif isinstance(child, ast.ClassDef):
                methods = [sub.name for sub in child.body if isinstance(sub, ast.FunctionDef)]
                structure.append(f"Class: {child.name} (Methods: {', '.join(methods)})")
            elif isinstance(child, ast.FunctionDef):
                args = [arg.arg for arg in child.args.args]
                structure.append(f"Function: {child.name}({', '.join(args)})")
                
        if not structure:
            return f"File '{path}' is parsed successfully, but contains no high-level class or function declarations."
        return f"--- AST Outline for {path} ---\n" + "\n".join(structure)
    except SyntaxError as se:
        return f"Syntax Error parsing '{path}': {str(se)}"
    except Exception as e:
        return f"Error parsing AST for '{path}': {str(e)}"

if __name__ == "__main__":
    mcp.run()
