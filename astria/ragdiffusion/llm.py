import requests
import json
import os
import re

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

TEST_FN = "test.txt"
test_pth = lambda fn=TEST_FN: Path(__file__).resolve().parent / 'prompts' / fn

OPEN_AI_URL = "https://api.openai.com/v1/chat/completions"
MAX_TRIES = 5

to_four_digits = lambda x: float(Decimal(x).quantize(Decimal('0.001'), ROUND_HALF_UP))

def generate_n_column_layout_regions(n, box_width_frac=0.8, box_height_frac=0.8):
    """
    Returns a dict for the JSON that places n bounding boxes side by side in one row.
    Each bounding box is centered horizontally in its "cell," and at the top 30% 
    vertically (just as an example).
    """
    col_width = 1.0 / n

    # 1) SR_hw_split_ratio: single row of height=1.0, plus n columns each = 1/n
    #    "1.0, (1/n), (1/n), ..., (1/n)"
    row_ratios = [col_width] * n
    sr_hw_split_ratio = ",".join(str(to_four_digits(x)) for x in row_ratios)

    # 2) For each subject i, compute:
    #    - m_offset (x offset)
    #    - m_scale (width fraction)
    #    - n_offset (y offset)
    #    - n_scale (height fraction)
    # We'll center the box_width_frac in the column. That means:
    #    column_x_start = i*col_width
    #    margin_x = (col_width - col_width*box_width_frac)/2
    #             = col_width*(1-box_width_frac)/2
    #    m_offset_i = column_x_start + margin_x
    #    m_scale_i  = col_width*box_width_frac
    margin_x = col_width * (1 - box_width_frac) / 2.0

    HB_m_offsets = []
    HB_n_offsets = []
    HB_m_scales  = []
    HB_n_scales  = []

    for i in range(n):
        this_x = i*col_width + margin_x
        HB_m_offsets.append(to_four_digits(this_x))
        HB_m_scales.append(to_four_digits(col_width * box_width_frac))

        HB_n_offsets.append(to_four_digits((1.0 - box_height_frac) / 2))
        HB_n_scales.append(box_height_frac)

    # Create the placeholders for prompts
    HB_prompts = [f"Subject_{i+1}" for i in range(n)]

    output = {
      "SR_hw_split_ratio": sr_hw_split_ratio,
      "SR_prompt": "Placeholder SR_prompt here. Use BREAK if you want multiple sub-prompts.",
      "HB_prompt_list": HB_prompts,
      "HB_m_offset_list": HB_m_offsets,
      "HB_n_offset_list": HB_n_offsets,
      "HB_m_scale_list": HB_m_scales,
      "HB_n_scale_list": HB_n_scales
    }
    return output


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


def openai_gpt4o_get_multi_lora_prompts(prompt, n, cache=False):
    # Cache for debugging
    if test_pth(f"{prompt[0:32]}.txt").exists() and cache:
        with open(test_pth(f"{prompt[0:32]}.txt"), 'r') as f:
            return json.load(f)

    api_key = os.environ.get('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError('Please provide an OpenAI API key via `OPENAI_API_KEY`')

    template_pth = Path(__file__).resolve().parent / 'prompts' / 'prompt_only_sr.txt'
    with open(template_pth, 'r',encoding="utf-8") as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n As a hint, there are **{n}** many people or objects in this image. Can you find all of them?. Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting for Step 1, then response with a markdown formatted JSON section for Step 2."

    textprompt= f"{' '.join(template)} \n {user_textprompt}"

    tries = 0
    regions = None
    while tries < MAX_TRIES and regions is None:
        try:
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
            response = requests.request("POST", OPEN_AI_URL, headers=headers, data=payload)
            # response_txt = response.text
            obj = response.json()
            text = obj['choices'][0]['message']['content']

            match = re.search(
                r'```json\s*(.*?)\s*```',
                text,
                re.DOTALL,
            )
            if match:
                json_raw_content = match.group(1)
                regions = json.loads(json_raw_content)
            else:
                raise json.JSONDecodeError("No matching JSON block detected")
            
            if 'HB_prompt_list' not in regions.keys():
                raise ValueError('missing key HB_prompt_list')
            if 'SR_prompt' not in regions.keys():
                raise ValueError('missing key SR_prompt')

        except json.JSONDecodeError as e:
            print(f"Failed to get parseable response ({str(e)}), retrying...")
            tries += 1
            continue
        except ValueError as e:
            print(f"Failed to get parseable response ({str(e)}), retrying...")
            tries += 1
            continue

    # Cache for debugging
    if cache:
        with open(test_pth(f"{prompt[0:32]}.txt"), 'w') as f:
            json.dump(regions, f, indent=2)

    return regions

def openai_gpt4o_get_regions(prompt, cache=False):
    # Cache for debugging
    if test_pth(f"{prompt[0:32]}.txt").exists() and cache:
        with open(test_pth(f"{prompt[0:32]}.txt"), 'r') as f:
            return json.load(f)

    api_key = os.environ.get('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError('Please provide an OpenAI API key via `OPENAI_API_KEY`')

    template_pth = Path(__file__).resolve().parent / 'prompts' / 'prompt.txt'
    with open(template_pth, 'r',encoding="utf-8") as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting for Step 1, then response with a markdown formatted JSON section for Step 2."

    textprompt= f"{' '.join(template)} \n {user_textprompt}"

    tries = 0
    regions = None
    while tries < MAX_TRIES and regions is None:
        try:
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
            response = requests.request("POST", OPEN_AI_URL, headers=headers, data=payload)
            # response_txt = response.text
            obj = response.json()
            text = obj['choices'][0]['message']['content']

            match = re.search(
                r'```json\s*(.*?)\s*```',
                text,
                re.DOTALL,
            )
            if match:
                json_raw_content = match.group(1)
                regions = json.loads(json_raw_content)
            else:
                raise json.JSONDecodeError("No matching JSON block detected")

        except json.JSONDecodeError as e:
            print(f"Failed to get parseable response ({str(e)}), retrying...")
            tries += 1
            continue

    # Cache for debugging
    if cache:
        with open(test_pth(f"{prompt[0:32]}.txt"), 'w') as f:
            json.dump(regions, f, indent=2)

    return regions


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

if __name__ == '__main__':
    print(json.dumps(generate_n_column_layout_regions(3), indent=2))