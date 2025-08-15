import os
import sys
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai_prompts import system_prompt
from call_functions import available_functions


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    parser = argparse.ArgumentParser()
    parser.add_argument('user_prompt')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    messages = [
    types.Content(role="user", parts=[types.Part(text=args.user_prompt)]),
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages, 
        config=types.GenerateContentConfig(
        tools=[available_functions], system_instruction=system_prompt
    )
    )
    if args.verbose:
        print("User prompt:", args.user_prompt)
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Calling function: {function_call.name}({function_call.args})")
    print("response:")
    print(response.text)


if __name__ == "__main__":
    main()
