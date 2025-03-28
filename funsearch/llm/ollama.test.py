import requests
import json
from pydantic import BaseModel


class OllamaResponse(BaseModel):
    new_function: str


class Demo:
    def _ask_llm(self, prompt: str) -> str:
        # URL for the generate endpoint of the ollama server.
        url = "http://ollama:11434/api/generate"

        # Payload for the POST request
        payload = {
            "prompt": prompt,
            "model": "gemma3:12b",
            "format": OllamaResponse.model_json_schema(),
            "stream": False,
            # You can include other model parameters as needed.
        }

        # Send the POST request to the generate endpoint.
        response = requests.post(url, json=payload)
        response.raise_for_status()

        # The API is expected to return a JSON response containing the generated text.
        result = response.json()

        # Try to parse the generated text as a JSON object.
        generated_text = result.get("response", "")
        try:
            parsed_output = json.loads(generated_text)
            new_function = parsed_output.get("new_function", "")
        except json.JSONDecodeError:
            # Fallback: If structured output parsing fails, return the raw generated text.
            new_function = generated_text

        return new_function


# Usage example:
if __name__ == "__main__":
    demo = Demo()
    prompt = '''"""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity. 
"""

import numpy as np
import scipy

#Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0]*MAX_NPARAMS

def equation_v0(x: np.ndarray, v: np.ndarray, params: np.ndarray):
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * x  +  params[1] * v  + params[2]
    return dv

def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Improved version of `equation_v0`.
    """
'''
    # Get the new function from the LLM via the generate endpoint.
    new_func = demo._ask_llm(prompt)
    print("Newly Generated Function:")
    print(new_func)
