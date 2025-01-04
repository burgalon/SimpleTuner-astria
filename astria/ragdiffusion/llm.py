import requests
import json
import os
import re

from pathlib import Path

TEST_FN = "test.txt"
test_pth = Path(__file__).resolve().parent / 'prompts' / TEST_FN


def parse_possible_list(value):
    """
    Attempt to parse `value` (which is presumably a string) as:
      1. A JSON list (or other JSON type)
      2. If #1 fails, a comma-separated list of floats/ints
      3. Otherwise just return the original string
    """
    # If not a string, we don't need to parse it.
    if not isinstance(value, str):
        return value
    
    # 1. Try to parse as JSON (e.g. "[1, 2, 3]" or "\"Hello\"" or "true")
    try:
        parsed_json = json.loads(value)
        return parsed_json
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 2. Try to parse as a comma-separated list of numeric values:
    #    e.g., "0.33, 0.34, 0.33" -> [0.33, 0.34, 0.33]
    parts = [x.strip() for x in value.split(",")]
    if len(parts) > 1:  # At least 2 items => likely comma-separated
        parsed_list = []
        all_numeric = True
        for part in parts:
            try:
                # Try float first
                num = float(part)
                parsed_list.append(num)
            except ValueError:
                all_numeric = False
                break
        if all_numeric:
            return parsed_list
    
    # 3. If all attempts fail, just return the original string
    return value


def openai_gpt4o_get_regions(prompt):
    # Cache for debugging
    # if test_pth.exists():
    #     with open(test_pth, 'r') as f:
    #         return json.load(f)
    
    api_key = os.environ.get('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError('Please provide an OpenAI API key via `OPENAI_API_KEY`')
    
    url = "https://api.openai.com/v1/chat/completions"

    template_pth = Path(__file__).resolve().parent / 'prompts' / 'prompt.txt'
    with open(template_pth, 'r',encoding="utf-8") as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting."
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": textprompt
            }
        ]
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # response_txt = response.text
    obj=response.json()
    text=obj['choices'][0]['message']['content']

    # Cache for debugging
    # with open(test_pth, 'w') as f:
    #     json.dump(get_params_dict(text), f, indent=2)

    return get_params_dict(text)


def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: (.*?)(?=\n|\Z)", response)
    if split_ratio_match:
        SR_hw_split_ratio = split_ratio_match.group(1)
        # print("Final split ratio:", final_split_ratio)
    else:
        SR_hw_split_ratio="NULL"
        # print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        SR_prompt = prompt_match.group(1).strip()
        # print("Regional Prompt:", regional_prompt)
    else:
        SR_prompt="NULL"
        # print("Regional Prompt not found.")

    HB_prompt_list_match = re.search(r"HB_prompt_list: (.*?)(?=\n|\Z)", response)
    if HB_prompt_list_match:
        HB_prompt_list = HB_prompt_list_match.group(1).strip()
        # print("sub_prompt_list:", sub_prompt_list)
    else:
        HB_prompt_list="NULL"
        # print("sub_prompt_list not found.")

    HB_m_offset_list_match = re.search(r"HB_m_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_m_offset_list_match:
        HB_m_offset_list = HB_m_offset_list_match.group(1).strip()
        # print("x_offset_list:", x_offset_list)
    else:
        HB_m_offset_list="NULL"
        # print("x_offset_list not found.")
    
    HB_n_offset_list_match = re.search(r"HB_n_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_n_offset_list_match:
        HB_n_offset_list = HB_n_offset_list_match.group(1).strip()
        # print("y_offset_list:", y_offset_list)
    else:
        HB_n_offset_list="NULL"
        # print("y_offset_list not found.")

    HB_m_scale_list_match = re.search(r"HB_m_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_m_scale_list_match:
        HB_m_scale_list = HB_m_scale_list_match.group(1).strip()
        # print("x_scale_list:", x_scale_list)
    else:
        HB_m_scale_list="NULL"
        # print("x_scale_list not found.")

    HB_n_scale_list_match = re.search(r"HB_n_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_n_scale_list_match:
        HB_n_scale_list = HB_n_scale_list_match.group(1).strip()
        # print("y_scale_list:", y_scale_list)
    else:
        HB_n_scale_list="NULL"
        # print("y_scale_list not found.")

    image_region_dict = {
        'SR_hw_split_ratio': SR_hw_split_ratio,
        'SR_prompt': parse_possible_list(SR_prompt),
        'HB_prompt_list': parse_possible_list(HB_prompt_list),
        'HB_m_offset_list': parse_possible_list(HB_m_offset_list),
        'HB_n_offset_list': parse_possible_list(HB_n_offset_list),
        'HB_m_scale_list': parse_possible_list(HB_m_scale_list),
        'HB_n_scale_list': parse_possible_list(HB_n_scale_list),
    }
    return image_region_dict