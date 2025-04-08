from google import genai
from funsearch import llmsr
from infra.ai import llm


def test_py_mutation_engine():
    gemini_client = genai.Client(api_key=llm.GOOGLE_CLOUD_API_KEY)
    engine = llmsr.PyMutationEngine(
        prompt_comment="",
        docstring="",
        gemini_client=gemini_client
    )
    response = engine._ask_gemini("1 + 1")
    # response = engine._ask_ollama("1 + 1")
    print(response)


if __name__ == '__main__':
    test_py_mutation_engine()
