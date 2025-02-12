import os
import json

def generate_settings(num_settings, settings_path):
    """
    Generates a list of negotiation settings.
    Each setting is a dictionary containing a scenario description and some numeric parameters.
    """
    settings = []
    for i in range(num_settings):
        # generate setting

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f_settings:
        for setting in settings:
            f_settings.write(json.dumps(setting) + "\n")
    print(f"Saved {len(settings)} settings to {settings_path}")
    return settings