import json
import os
import re
from typing import Any

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file

    Args:
        data: Data to be saved (must be JSON serializable)
        filepath: Save path including filename
        indent: JSON indentation format, defaults to 2
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(f"Data saved to: {filepath}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def preprocess_response_string(response_text: str) -> str:
    if response_text.startswith('```json') and response_text.endswith('```'):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith('```') and response_text.endswith('```'):
        response_text = response_text[3:-3].strip()
    response_text = response_text.replace("```", "").replace("json", "").strip()
    # Remove trailing commas
    response_text = re.sub(r',\s*}', '}', response_text)
    response_text = re.sub(r',\s*]', ']', response_text)
    return response_text


