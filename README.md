# FastContext AI Subagent Demo

A lightweight, local implementation demonstrating the core mechanics of the **FastContext** architecture for autonomous AI coding agents. 

Rather than polluting an LLM's context window by loading entire files, this subagent navigates a codebase using read-only native Linux utilities (`glob`, `grep`, and `sed`). It explores dynamically over multiple turns and returns only the highly specific line ranges required to answer the user's query.

## 🚀 Features

*   **Native Tool Calling:** Orchestrates local bash commands (`glob` for path discovery, `grep -i -n` for regex searching, and `sed` for targeted reading) safely via Python's `subprocess`.
*   **Multi-Turn Exploration:** The LLM autonomously loops through observations, deciding when to search broader directories versus looking deeper inside specific scripts.
*   **Schema Hallucination Recovery:** Built-in Python fallbacks to handle common LLM tool-calling quirks (e.g., passing `'files'` instead of the strict JSON schema `'file_path'`).
*   **Strict Output Formatting:** Enforces the `<final_answer>` XML-style citation block required to cleanly pass evidence back to a primary coding agent.

## 📂 The Testbed

This repository is designed to test how well an AI agent routes context across distinct machine learning domains. Inside the `testbed/` directory, you will find sample scripts converted from Jupyter Notebooks representing complex ML pipelines:

*   **`testbed/vision.py`:** Computer vision pipelines (e.g., EfficientNet initializations and image augmentations).
*   **`testbed/text.py`:** Natural language processing architectures (e.g., DeBERTa configurations, LoRA fine-tuning, and tokenization logic).
*   **`testbed/speech.py`:** Audio processing and speech recognition pipelines (e.g., loading and processing `.wav` datasets).

The subagent is challenged to figure out *which* file holds the relevant logic without being explicitly told.

## 🛠️ Setup & Installation

**1. Install Dependencies**
This project requires the Hugging Face Hub client and `python-dotenv` for secure credential management.
```bash
pip install huggingface_hub python-dotenv
