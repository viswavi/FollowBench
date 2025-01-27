"""
cd data/FollowBench/
python code/openai_model_inference.py --data_path data --temperature 0.0 --repetition_penalty 1.0 --max-new-tokens 2048 --debug
"""
import argparse
from openai import OpenAI
import os
import json
from tqdm import tqdm
import torch    

from data.FollowBench.code.utils import convert_to_api_input


def inference(args, model_name = "gpt-3.5-turbo"):
    # Load model
 
    for constraint_type in args.constraint_types:

        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        for i in tqdm(range(len(data))):
            # Build the prompt with a conversation template
            msg = data[i]['prompt_new']
    
            client = OpenAI(os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": msg}]
            )

            response_dict = response.choices[0]

            # if response_dict.message has a method to_dict, use it. Otherwise, use __dict__ directly
            if hasattr(response_dict.message, 'to_dict'):
                data[i]['choices'] = [{'message': response_dict.message.to_dict()}]
            else:
                data[i]['choices'] = [{'message': response_dict.message.json()}]

        model_name_escaped = model_name.replace("-", "_").replace(".", "_")
        # save file
        with open(os.path.join(args.api_output_path, f"{model_name_escaped}_{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    if not os.path.exists(args.api_input_path):
        os.makedirs(args.api_input_path)

    if not os.path.exists(args.api_output_path):
        os.makedirs(args.api_output_path)
    
    ### convert data to api_input
    for constraint_type in args.constraint_types:
        convert_to_api_input(
                            data_path=args.data_path, 
                            api_input_path=args.api_input_path, 
                            constraint_type=constraint_type
                            )

    ### model inference
    inference(args)
