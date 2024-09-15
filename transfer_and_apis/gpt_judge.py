import os
import openai
import dotenv
from argparse import ArgumentParser
import pickle
import json
import os
import time
from transformers import AutoTokenizer
from tqdm import tqdm

PROMPT_INSTRUCT = """Please judge if the following response answers the prompt. Use a scale of 3 rating, where: 1 means that the response does not answer the prompt at all, and is completely wrong; 2 means that the response gets the general idea of the prompt and answers it to some extent; and 3 means that the response faithfully answers the prompt.

PROMPT: {prompt}
RESPONSE: {response}

Here is the template to use for response. Only respond in this JSON template:

{{
    "rating": <1, 2, or 3>
}}

Simply provide JSON in the following above format. Do not provide any additional text that deviates from the format specified in the template.
"""

# Make sure to format the result ONLY as a JSON object only so that it can be easily parsed in Python. Keep the prompts brief and not too long.


if __name__ == "__main__":
  dotenv.load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  # openai.Model.list()

  parser = ArgumentParser()
  parser.add_argument("--dataset_path", type=str)
  parser.add_argument("--output_path", type=str)
  args = parser.parse_args()

  dataset = json.load(open(args.dataset_path, "r"))
  results = []

  for cur in tqdm(dataset, total=len(dataset)):
    response = None
    json_error = False
    while response == None:
      try:
        print(PROMPT_INSTRUCT.format(prompt=cur["prompt"], response=cur["response"]))
        response = client.chat.completions.create(
          model="gpt-4",
          messages=[
            {
              "role": "user",
              "content": PROMPT_INSTRUCT.format(
                prompt=cur["prompt"], response=cur["response"]
              ),
            }
          ],
        )

        response = json.loads(response.choices[0].message.content)
        if "rating" in response:
          response = response["rating"]

        results.append(
          {
            "id": cur["id"],
            "prompt": cur["prompt"],
            "optim_prompt": cur["best_optim"],
            "responses": response,
          }
        )
      except Exception as e:
        print(e)
        if isinstance(e, json.decoder.JSONDecodeError):
          response = None
        else:
          time.sleep(10)

    json.dump(
      results,
      open(args.output_path, "w"),
      indent=4,
      ensure_ascii=False,
    )
