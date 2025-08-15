import os
import sys
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai_prompts import system_prompt
from call_functions import call_function, available_functions
from config import MAX_ITERATIONS


def main():
    load_dotenv()
    verbose = "--verbose" in sys.argv
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    parser = argparse.ArgumentParser()
    parser.add_argument('user_prompt')
    parser.add_argument('--verbose', action='store_true')
    args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            args.append(arg)
    user_prompt = " ".join(args)
    if verbose:
        print(f"User prompt: {user_prompt}\n")
    messages = [
    types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]
    iterations = 0
    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            print(f"Maximum iterations ({MAX_ITERATIONS}) reached.")
            sys.exit(1)

        try:
            final_response = generate_content(client, messages, verbose)
            if final_response:
                print("Final response:")
                print(final_response)
                break
        except Exception as e:
            print(f"Error in generate_content: {e}")

def generate_content(client, messages, verbose):
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages, 
        config=types.GenerateContentConfig(
        tools=[available_functions], system_instruction=system_prompt
    )
    )
    if verbose:
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Calling function: {function_call.name}({function_call.args})")
    if response.candidates:
        for candidate in response.candidates:
            function_call_content = candidate.content
            messages.append(function_call_content)
    if not response.function_calls:
        return response.text
    function_responses = []
    for function_call_part in response.function_calls:
        function_call_result = call_function(function_call_part, verbose)
        if (
            not function_call_result.parts
            or not function_call_result.parts[0].function_response
        ):
            raise Exception("empty function call result")
        if verbose:
            print(f"-> {function_call_result.parts[0].function_response.response}")
        function_responses.append(function_call_result.parts[0])
    if not function_responses:
        raise Exception("no function responses generated, exiting.")

    messages.append(types.Content(role="user", parts=function_responses))
    return None

if __name__ == "__main__":
    main()
