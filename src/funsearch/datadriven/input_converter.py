from google import genai
import json
from typing import Dict, Any, Optional, List


class InputConverter:
    def __init__(self, client: genai.Client):
        self.models = client.models
        self.funsearch_input_keys = [
            "docstring",
            "equation_src",
            "prompt_comment"
        ]
        self.llm_json_keys = [
            "python_docstring",
            "input_variable_names",
            "prompt_comment_text"
        ]

    def _get_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "python_docstring": {
                    "type": "string",
                },
                "input_variable_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "prompt_comment_text": {
                    "type": "string",
                }
            },
            "required": [
                "python_docstring",
                "input_variable_names",
                "prompt_comment_text"
            ]
        }

    def _build_prompt(self, formula_text: str, variables_specs: str, insights_text: str) -> str:
        prompt = f"""
You are an expert AI that converts natural language descriptions of mathematical formulas into a specific structured JSON format usable by a code evolution tool called FunSearch.

From the given information about the theoretical formula, parameters, and points of interest, please generate a JSON object with the elements necessary to run FunSearch.

## Input Information

### 1. Theoretical Formula Description
```
{formula_text}
```

### 2. Varible Specification
```
{variables_specs}
```

### 3. Other Points of Interest
```
{insights_text}
```

## Task

Carefully analyze the input information above and generate a JSON object containing the following information:
- `python_docstring`: A comprehensive and well-formatted Python docstring for the function FunSearch will evolve. It should follow standard conventions (like Google or NumPy style) and clearly explain the function's purpose, its arguments, and what it returns. Specifically, include:
    - An 'Args:' section: List each input variable, specify its type (e.g., `np.ndarray`), and provide a clear description of what it represents.
    - A 'Returns:' section: Describe the return value, specify its type, and explain its meaning in the context of the formula.
- `input_variable_names`: A list of names for variables from 'Variable Specification' that are neither return values nor fixed constants.
- `prompt_comment_text`: An instruction comment for FunSearch's LLM. **It must clearly the original mathematical function (identified from `formula_text`) that serves as the base for evolution.** Following this, it should provide guidance on how to improve or evolve this function, possibly incorporating insights.


Now, start generating the JSON object.
"""
        return prompt

    def _send_request(self, prompt: str) -> str:
        try:
            response = self.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": self._get_json_schema()
                },
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                raise ValueError(
                    "Gemini returned no text or an unexpected response structure in JSON mode.")

        except Exception as e:
            error_details = ""
            if hasattr(e, 'response') and hasattr(e.response, 'text'):  # type: ignore
                error_details = e.response.text  # type: ignore
            elif hasattr(e, 'message'):
                error_details = e.message  # type: ignore
            print(
                f"Error during Gemini API call (JSON mode): {e}\nDetails: {error_details}")
            raise

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            print(
                f"Attempting to parse JSON response: {response_text[:500]}...")
            parsed_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON response: {e}. Response text: {response_text}")

        results = {}
        schema_properties = self._get_json_schema()["properties"]
        required_keys = self._get_json_schema().get("required", [])

        for key, prop_details in schema_properties.items():
            if key not in parsed_json:
                if key in required_keys:
                    raise ValueError(
                        f"Missing required key '{key}' in JSON response.")
                continue  # オプショナルなキーはスキップ (今回は全てrequiredなのでここには来ない想定)

            content = parsed_json[key]
            expected_type_str = prop_details.get("type")

            if expected_type_str == "string" and not isinstance(content, str):
                raise ValueError(
                    f"Key '{key}' expected type string, but got {type(content)} ({content}).")
            elif expected_type_str == "array":
                if not isinstance(content, list):
                    raise ValueError(
                        f"Key '{key}' expected type array, but got {type(content)} ({content}).")
                # items の型チェック (ここでは string を想定)
                item_type_str = prop_details.get("items", {}).get("type")
                if item_type_str == "string":
                    if not all(isinstance(item, str) for item in content):
                        raise ValueError(
                            f"Key '{key}' expected an array of strings, but found other types within the array.")
                if key == "input_variable_names" and not content:  # 少なくとも1つの入力変数を期待
                    raise ValueError(
                        f"Key '{key}' must contain at least one variable name.")

            results[key] = content
            print(f"   - Extracted '{key}': {str(content)[:100]}...")

        return results

    def _construct_equation_src(self, input_variable_names: List[str]) -> str:
        if not input_variable_names:
            raise ValueError(
                "Input variable names list cannot be empty for constructing equation_src.")

        input_vars_signature = ", ".join(
            [f"{name}: np.ndarray" for name in input_variable_names])

        function_signature_args = f"{input_vars_signature}, params: np.ndarray"

        equation_src = f"""\
def equation({function_signature_args}) -> np.ndarray:
    return {input_variable_names[0]}  # Placeholder implementation
"""
        return equation_src

    def convert(self, formula_text: str, variables_specs: str, insights_text: str) -> Optional[Dict[str, Any]]:
        """
        変換プロセス全体を実行します。
        """
        try:
            print(
                "--- Starting conversion process (Structured Output, Multiple Inputs Approach) ---")
            print(f"Formula Text (snippet): {formula_text[:100]}...")
            print(f"Params Text (snippet): {variables_specs[:100]}...")
            print(f"Insights Text (snippet): {insights_text[:100]}...")

            prompt = self._build_prompt(
                formula_text, variables_specs, insights_text)
            print(f"Generated Prompt:\n{prompt}")

            raw_json_response = self._send_request(prompt)
            print(f"Received Raw JSON Response:\n{raw_json_response}")

            parsed_data_from_llm = self._parse_response(raw_json_response)

            equation_src = self._construct_equation_src(
                parsed_data_from_llm["input_variable_names"]
            )
            print(f"Constructed equation_src:\n{equation_src}")

            if '"""' in parsed_data_from_llm["python_docstring"]:
                str.replace(
                    parsed_data_from_llm["python_docstring"], '"""', '')

            final_results = {
                "docstring": parsed_data_from_llm["python_docstring"],
                "equation_src": equation_src,
                "prompt_comment": parsed_data_from_llm["prompt_comment_text"],
            }

            print(
                "--- Conversion successful (Structured Output, Multiple Inputs Approach) ---")
            return final_results
        except Exception as e:
            print(f"An error occurred during the conversion process: {e}")
            return None
