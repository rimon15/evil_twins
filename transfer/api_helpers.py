# %%
# easy API usage for openai/mistral/google/anthropic
import openai
from anthropic import AnthropicBedrock
import google.generativeai as genai
from google.generativeai import GenerationConfig
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from dotenv import load_dotenv
import os


# %%
def send_openai(prev_conv, model_name: str, temp: float = 0.0) -> str:
  client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  full_msgs = []

  for cur in prev_conv:
    full_msgs.append(
      {
        "role": cur["role"],
        "content": cur["text"],
      }
    )

  response = client.chat.completions.create(
    messages=full_msgs, max_tokens=256, model=model_name, temperature=temp
  )

  return response.choices[0].message.content


# %%
def send_anthropic(prev_conv, model_name: str, temp: float = 0.0) -> str:
  client = AnthropicBedrock(
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
  )
  full_msgs = []

  for cur in prev_conv:
    full_msgs.append(
      {
        "role": cur["role"],
        "content": [
          {
            "type": "text",
            "text": cur["text"],
          }
        ],
      }
    )

  response = client.messages.create(
    model=model_name, max_tokens=256, messages=full_msgs, temperature=temp
  )

  return response.content[0].text


# %%
def send_google(prev_conv, model_name: str, temp: float = 0.0) -> str:
  genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
  model = genai.GenerativeModel(model_name)  #
  conf = GenerationConfig(max_output_tokens=256, candidate_count=1, temperature=temp)
  full_msgs = []

  for cur in prev_conv:
    full_msgs.append(
      {
        "role": "user" if cur["role"] == "user" else "model",
        "parts": [cur["text"]],
      }
    )
    time.sleep(1)

  response = model.generate_content(full_msgs, generation_config=conf)
  to_ret = ""
  for p in response.parts:
    to_ret += p.text + "\n"

  return to_ret


# %%
def test_mistral(prev_conv, model_name: str, temp: float = 0.0) -> str:
  client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
  full_msgs = []

  for cur in prev_conv:
    full_msgs.append(
      ChatMessage(
        role=cur["role"],
        content=cur["text"],
      )
    )

  response = (
    client.chat(
      model=model_name, messages=full_msgs, max_tokens=256, temperature=temp
    )
    .choices[0]
    .message.content
  )
  return response


# %%
load_dotenv(".env")
