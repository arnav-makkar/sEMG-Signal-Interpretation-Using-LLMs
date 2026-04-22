#!/usr/bin/env python3
"""
Muscle Activation Prediction
==============================
Loads activation_prompt.txt for each participant, calls LLM API,
and saves the structured JSON response to activation_result.json.

Supports: --model gemini (default) | --model groq

Output: DATA/{P}/activation_result.json
"""

import os
import re
import json
import sys
from pathlib import Path

PARTICIPANTS = [f'P{i}' for i in range(1, 11)]  # P1-P10
DATA_DIR = 'DATA'
API_KEY_FILE = '.env'
TEMPERATURE = 0.1
MAX_TOKENS = 1800
MAX_RETRIES = 2

# Model configs
MODELS = {
    'gemini': 'gemini-3.1-flash-lite-preview',
    'groq':   'llama-3.1-8b-instant',
}


# ==============================
# API KEY LOADING
# ==============================

def load_env_file(path=".env"):
    env_vars = {}
    env_path = Path(path)
    if not env_path.is_file():
        return env_vars
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars


def get_api_key(key_name):
    api_key = os.environ.get(key_name)
    if not api_key:
        env_vars = load_env_file(API_KEY_FILE)
        api_key = env_vars.get(key_name)
    return api_key


# ==============================
# GEMINI CLIENT
# ==============================

def get_gemini_client():
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = get_api_key('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    return client


def predict_gemini(participant, client, model_name):
    prompt_path = os.path.join(DATA_DIR, participant, 'activation_prompt.txt')
    output_path = os.path.join(DATA_DIR, participant, 'activation_result.json')

    if not os.path.exists(prompt_path):
        print(f"  ERROR: {prompt_path} not found.")
        return False

    with open(prompt_path) as f:
        full_prompt = f.read()

    # Split system / user
    if 'USER:\n' in full_prompt:
        parts = full_prompt.split('USER:\n', 1)
        system_text = parts[0].replace('SYSTEM:\n', '').strip()
        user_text = parts[1].strip()
    else:
        system_text = "You are a clinical EMG analysis assistant. Return STRICT JSON only."
        user_text = full_prompt

    from google.genai import types

    print(f"  Calling Gemini API for {participant}...")

    result = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_text,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            raw_text = response.text

            # Token usage
            usage_meta = response.usage_metadata
            prompt_tokens = getattr(usage_meta, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(usage_meta, 'candidates_token_count', 0) or 0
            print(f"  Tokens: {prompt_tokens} prompt + {completion_tokens} completion")

            result = extract_json_from_response(raw_text)
            if 'parse_error' not in result:
                break
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt+1}/{MAX_RETRIES}: JSON parse failed, retrying...")

        except Exception as e:
            print(f"  API ERROR for {participant} (attempt {attempt+1}): {e}")
            if attempt >= MAX_RETRIES:
                return False

    if result is None:
        return False

    result['_meta'] = {
        'participant': participant,
        'model': model_name,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'parse_success': 'parse_error' not in result,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return _print_result(result, output_path)


# ==============================
# GROQ CLIENT
# ==============================

def get_groq_client():
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq not installed. Run: pip install groq")
        sys.exit(1)

    api_key = get_api_key('GROQ_API_KEY')
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file")
        sys.exit(1)

    return Groq(api_key=api_key)


def predict_groq(participant, client, model_name):
    prompt_path = os.path.join(DATA_DIR, participant, 'activation_prompt.txt')
    output_path = os.path.join(DATA_DIR, participant, 'activation_result.json')

    if not os.path.exists(prompt_path):
        print(f"  ERROR: {prompt_path} not found.")
        return False

    with open(prompt_path) as f:
        full_prompt = f.read()

    if 'USER:\n' in full_prompt:
        parts = full_prompt.split('USER:\n', 1)
        system_text = parts[0].replace('SYSTEM:\n', '').strip()
        user_text = parts[1].strip()
    else:
        system_text = "You are a clinical EMG analysis assistant. Return STRICT JSON only."
        user_text = full_prompt

    print(f"  Calling Groq API for {participant}...")

    result = None
    usage = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user",   "content": user_text},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content
            usage = response.usage
            print(f"  Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")

            result = extract_json_from_response(raw_text)
            if 'parse_error' not in result:
                break
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt+1}/{MAX_RETRIES}: JSON parse failed, retrying...")

        except Exception as e:
            print(f"  API ERROR for {participant} (attempt {attempt+1}): {e}")
            if attempt >= MAX_RETRIES:
                return False

    if result is None:
        return False

    result['_meta'] = {
        'participant': participant,
        'model': model_name,
        'prompt_tokens': usage.prompt_tokens if usage else 0,
        'completion_tokens': usage.completion_tokens if usage else 0,
        'parse_success': 'parse_error' not in result,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return _print_result(result, output_path)


# ==============================
# SHARED
# ==============================

def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {'parse_error': True, 'raw_response': text}


def _print_result(result, output_path):
    parse_ok = 'parse_error' not in result
    print(f"  {'OK' if parse_ok else 'PARSE FAILED'} -> {output_path}")
    if parse_ok:
        oa = result.get('overall_activation', '?')
        tp = result.get('temporal_profile', '?')
        ds = result.get('dominant_sensors', [])
        conf = result.get('confidence', '?')
        print(f"    overall_activation={oa} | profile={tp} | dominant={ds} | confidence={conf}")
    return parse_ok


def main():
    # Parse --model flag
    backend = 'gemini'  # default
    participants = PARTICIPANTS

    args = sys.argv[1:]
    if '--model' in args:
        idx = args.index('--model')
        if idx + 1 < len(args):
            backend = args[idx + 1].lower()
            args = args[:idx] + args[idx+2:]
        else:
            print("ERROR: --model requires a value (gemini or groq)")
            sys.exit(1)

    if args:
        participants = [args[0]]

    if backend not in MODELS:
        print(f"ERROR: Unknown model backend '{backend}'. Use: gemini or groq")
        sys.exit(1)

    model_name = MODELS[backend]
    print(f"Running activation prediction for: {participants}")
    print(f"Backend: {backend} | Model: {model_name} | temp={TEMPERATURE} | max_tokens={MAX_TOKENS}\n")

    # Init client
    if backend == 'gemini':
        client = get_gemini_client()
        predict_fn = predict_gemini
    else:
        client = get_groq_client()
        predict_fn = predict_groq

    success = 0
    for p in participants:
        print(f"\n{'='*50}")
        print(f"Participant: {p}")
        print(f"{'='*50}")
        if predict_fn(p, client, model_name):
            success += 1

    print(f"\n{'='*50}")
    print(f"Done: {success}/{len(participants)} successful")


if __name__ == '__main__':
    main()
