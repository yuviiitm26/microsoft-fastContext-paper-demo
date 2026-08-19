import os
import subprocess
import glob as pyglob
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from the .env file
load_dotenv()

class FastContextTools:
    """Implements the read-only tools using native Linux utilities."""
    
    @staticmethod
    def glob(pattern, directory="."):
        """Glob: Fast file pattern matching."""
        return pyglob.glob(f"{directory}/{pattern}", recursive=True)

    @staticmethod
    def grep(pattern, file_path):
        """Grep: Regex search over repository text using Linux grep."""
        try:
            # Added '-i' for case-insensitive matching
            result = subprocess.run(
                ['grep', '-i', '-n', pattern, file_path],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return [] 

    @staticmethod
    def read(file_path, offset=1, limit=50):
        """Read: Fetches file contents using sed."""
        try:
            end_line = offset + limit - 1
            result = subprocess.run(
                ['sed', '-n', f'{offset},{end_line}p', file_path],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')
            return [f"{i+offset}: {line}" for i, line in enumerate(lines)]
        except subprocess.CalledProcessError as e:
            return [f"Error reading file: {e}"]

class FastContextSubagent:
    def __init__(self):
        self.tools = FastContextTools()
        
        # Safely load the token from the environment
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Please ensure your .env file is set up correctly.")

        # Using Llama 3.3 for its excellent tool-calling capabilities
        self.client = InferenceClient(
            model="meta-llama/Llama-3.3-70B-Instruct",
            token=token
        )
        
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Fast file pattern matching tool. Supports glob patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
                            "directory": {"type": "string", "description": "Target directory"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Regex search over repository text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern to search for"},
                            "file_path": {"type": "string", "description": "Exact file to search in"}
                        },
                        "required": ["pattern", "file_path"]
                    }
                }
            }
        ]

    def explore(self, query):
        print(f"🔍 Asking LLM to analyze query: '{query}'")
        
        # Stricter system prompt to force correct tool usage
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a FastContext Codebase Exploration Agent. "
                    "Rule 1: You MUST use 'glob' first to discover real file paths. "
                    "Rule 2: You CANNOT pass wildcards (like **/*.py) into 'grep'. You must only pass exact, single file paths returned by glob. "
                    "Rule 3: Once you have used grep on the real files and found the context, stop exploring and return."
                )
            },
            {"role": "user", "content": query}
        ]

        citations = []
        max_turns = 8 # Safety bound for the exploration loop

        for turn in range(max_turns):
            response = self.client.chat_completion(
                messages=messages,
                tools=self.tool_definitions,
                max_tokens=500
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            if not response_message.tool_calls:
                print("✅ LLM finished exploring.")
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"🤖 Turn {turn + 1} - LLM decided to execute: {function_name} with arguments {arguments}")
                
                tool_result = ""
                if function_name == "glob":
                    result = self.tools.glob(arguments['pattern'], arguments.get('directory', '.'))
                    tool_result = str(result)
                    print(f"📂 Found: {result}")
                
                elif function_name == "grep":
                    # Fallback mapping in case the LLM hallucinates parameter names
                    target_file = arguments.get('file_path') or arguments.get('files') or arguments.get('file')
                    
                    if isinstance(target_file, list):
                        target_file = target_file[0]

                    if target_file:
                        matches = self.tools.grep(arguments['pattern'], target_file)
                        tool_result = str(matches)
                        
                        if matches:
                            line_num = int(matches[0].split(':')[0])
                            print(f"🔎 Found match at line {line_num}. Generating citation...")
                            citations.append({
                                "path": target_file,
                                "range": f"{line_num}-{line_num + 10}",
                                "note": f"Result of searching for {arguments['pattern']}"
                            })
                        else:
                            print(f"🔎 No matches found for {arguments['pattern']} in {target_file}.")
                    else:
                        tool_result = "Error: LLM did not provide a valid file target."
                        print("⚠️ LLM didn't provide a valid file target.")

                messages.append({
                    "role": "tool", 
                    "name": function_name, 
                    "content": tool_result, 
                    "tool_call_id": tool_call.id
                })
        
        return self._format_final_answer(citations)

    def _format_final_answer(self, citations):
        """Returns the compact file-and-line citations block."""
        output = "\n<final_answer>\n"
        for citation in citations:
            note = f" ({citation['note']})" if citation['note'] else ""
            output += f"{citation['path']}: {citation['range']}{note}\n"
        output += "</final_answer>"
        return output

if __name__ == "__main__":
    subagent = FastContextSubagent()
    
    # Updated query reflecting the actual models present in your notebooks
    test_query = (
        "Use glob to search the 'testbed' directory for python files. "
        "Find exactly where the EfficientNet model is initialized, "
        "where the DeBERTa model is loaded with LoRA, and how the .wav files are processed."
    )
    
    result = subagent.explore(test_query)
    print(result)