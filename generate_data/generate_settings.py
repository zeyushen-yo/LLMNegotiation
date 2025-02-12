import os
import json
from gpt import get_output, extract_from_text
from prompts import generate_setting_prompt

def generate_settings(num_settings, settings_path, model, max_tokens, agents):
    settings = []
    for i in range(num_settings):
        output = get_output(generate_setting_prompt.format(agent1=agents[0], agent2=agents[1]), model, max_tokens)
        setting = extract_from_text(output, "Answer:")

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f_settings:
        for setting in settings:
            f_settings.write(json.dumps(setting) + "\n")
    print(f"Saved {len(settings)} settings to {settings_path}")