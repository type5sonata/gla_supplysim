"""Saving and loading named scenario presets."""

import os
import json

PRESETS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'presets.json')


def load_presets_from_file():
    """Load all saved presets from file"""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_presets_to_file(presets):
    """Save all presets to file"""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)

def save_preset(name, thread_params, active_threads):
    """Save current configuration as a preset"""
    presets = load_presets_from_file()
    presets[name] = {
        'threads': active_threads,
        'params': thread_params
    }
    save_presets_to_file(presets)

def delete_preset(name):
    """Delete a preset by name"""
    presets = load_presets_from_file()
    if name in presets:
        del presets[name]
        save_presets_to_file(presets)

