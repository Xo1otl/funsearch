import requests
import json
from pydantic import BaseModel


class OllamaResponse(BaseModel):
    improved_equation: str


class Demo:
    def _ask_llm_chat(self, prompt: str) -> str:
        # URL for the generate endpoint of the ollama server.
        url = "http://ollama:11434/api/chat"

        payload = {
            "model": "gemma3:12b",
            "format": OllamaResponse.model_json_schema(),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        # Send the POST request to the generate endpoint.
        response = requests.post(url, json=payload)
        response.raise_for_status()

        # The API is expected to return a JSON response containing the generated text.
        result = response.json()

        # Try to parse the generated text as a JSON object.
        generated_text = result.get("message", {}).get("content", "")
        try:
            parsed_output = json.loads(generated_text)
            new_function = parsed_output.get("improved_function", "")
        except json.JSONDecodeError:
            # Fallback: If structured output parsing fails, return the raw generated text.
            new_function = generated_text

        return new_function

    def _ask_llm_generate(self, prompt: str) -> str:
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
        generated_text = result["response"]
        try:
            parsed_output = json.loads(generated_text)
            improved_equation = parsed_output.get("improved_equation", "")
        except json.JSONDecodeError:
            # Fallback: If structured output parsing fails, return the raw generated text.
            improved_equation = generated_text

        return improved_equation


# Usage example:
if __name__ == "__main__":
    demo = Demo()
    prompt = '''
You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. Complete the 'equation' function below, considering the physical meaning and relationships of inputs.
        

"""
Find the mathematical function skeleton that represents SHG efficiency in QPM devices.
"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0] * MAX_NPARAMS


def equation_v0(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Total domain length
    L = params[0] * width
    # Phase mismatch calculation
    delta_k = params[1]/wavelength + params[2] - np.pi/width
    # SHG efficiency using sinc^2 formula
    arg = delta_k * L / 2
    efficiency = params[3] * L**2 * np.sin(arg)**2 / arg**2
    return efficiency


# Improved version of `equation_v0`.
def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ 
    Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """

'''
    # Get the new function from the LLM via the generate endpoint.
    new_func = demo._ask_llm_chat(prompt)
    print("Newly Generated Function by chat:")
    print(new_func)

    new_func = demo._ask_llm_generate(prompt)
    print("Newly Generated Function by generate:")
    print(new_func)
