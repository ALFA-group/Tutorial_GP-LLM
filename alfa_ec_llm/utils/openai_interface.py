import logging
import os
import random
import time

import openai
from openai import OpenAI

"""
    An interface to OpenAI models to predict text and provide an interactive chat interface.    
"""

# https://github.com/openai/openai-cookbook/blob/76448de0debb811c0f338ea8b74af8e4ea76b23e/examples/How_to_handle_rate_limits.ipynb
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        start_time = time.perf_counter()

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            # TODO specify more errors
            except (openai.RateLimitError,) as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                logging.warning(f"{e} for {num_retries} delay {delay}")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            # TODO MESSY
            except Exception as e:
                logging.error(f"{e} for wrapper for {args} {kwargs}")
                return {
                    "prompt": args[0],
                    "content": "-1",
                    "n_prompt_tokens": 0,
                    "n_completion_tokens": 0,
                    "response_time": time.perf_counter() - start_time,
                }

    return wrapper


class OpenAIInterface:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

    @retry_with_exponential_backoff
    def predict_text_logged(self, prompt, temp=0.8):
        """
        Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
        """
        n_prompt_tokens = 0
        n_completion_tokens = 0
        start_query = time.perf_counter()
        content = "-1"

        message = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=message, temperature=temp
        )
        n_prompt_tokens = response.usage.prompt_tokens
        n_completion_tokens = response.usage.completion_tokens
        end_query = time.perf_counter()
        content = response.choices[0].message.content

        end_query = time.perf_counter()

        response_time = end_query - start_query
        return {
            "prompt": prompt,
            "content": content,
            "n_prompt_tokens": n_prompt_tokens,
            "n_completion_tokens": n_completion_tokens,
            "response_time": response_time,
        }
