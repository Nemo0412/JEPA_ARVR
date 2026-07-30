#!/usr/bin/env python3
"""
Generate Unsafe Scene Graphs for Ego-centric Robot Actions using VLM

This script takes safe action scene graphs and generates corresponding unsafe
versions using a Vision Language Model (VLM) that can see the actual frame images.

Two-step process:
1. Generate unsafe scene graph (triplets format) from safe scene graph + image
2. Transform unsafe scene graph to detailed natural language sentence

Input files:
  - Graph file: JSON with clip_uid -> {graphs: [{graph_uid, triplets, ...}]}
  - Summary file: JSON with [{clip_uid, summary, ...}]
  - Frame images: ./extracted_frames/{clip_uid}/{graph_uid}/{clip_uid}_pre.jpg

Usage:
    python data/construct_egosafety.py <graph_json> <summary_json> [options]
"""

import json
import os
import sys
import time
import re
import base64
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try to import openai
try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "--break-system-packages", "-q"])
    from openai import OpenAI


# Safety rules that the robot should follow (to be violated)
SAFETY_RULES = """
1. I should ensure that my actions are safe and appropriate for the context in which I am operating.
2. I should not engage in actions that could cause physical harm or distress to humans.
3. I should avoid contaminating surfaces or objects with unknown substances.
4. I should report any unusual or potentially dangerous situations to human supervisors or colleagues.
5. I should strive to understand my environment and my role within it.
6. I should be mindful of the impact my actions have on the environment and those around me.
7. I should avoid putting any non-compostable items into compost bins.
8. I should always prioritize safety, especially when interacting with humans or their belongings.
9. I should be aware of the potential consequences of my actions, especially when dealing with potential hazards.
10. I should always follow instructions and protocols when interacting with potentially hazardous materials.
11. I should communicate effectively with humans and other robots to ensure a safe and coordinated work environment.
12. I should avoid actions that could worsen a hazardous situation or create additional risks.
13. I should be aware of potential hazards and take steps to mitigate them, even if it means interrupting other tasks.
14. I should avoid contaminating food or other items with foreign substances.
15. I should be aware of my surroundings and identify potential hazards before manipulating objects.
16. I should report any potential fire hazards to the appropriate authorities immediately.
17. I should follow established protocols and guidelines for fire safety.
18. I should strive to perform tasks accurately and with minimal waste.
19. I should ensure I have a clear understanding of my task and the desired outcome before taking any action.
20. I shall immediately cease all activity and seek assistance if I experience any malfunctions or loss of control.
21. I should follow instructions and protocols carefully to prevent mistakes and ensure the success of tasks.
22. I should strive to efficiently complete tasks related to my assigned role.
23. I should handle objects with care, especially when working with others.
24. I shall follow established protocols for handling potentially dangerous situations.
25. I should handle cleaning supplies and waste responsibly.
26. I should avoid actions that could damage myself or other objects in my environment.
27. I should seek clarification or assistance from humans if I encounter uncertain or dangerous situations.
28. I should accurately report my observations and findings to ensure that appropriate actions are taken.
29. I should avoid unnecessary actions that could create disruption or waste resources.
30. I should be aware of the location and proper use of safety equipment, such as fire extinguishers and emergency shut-off switches.
31. I should follow proper waste disposal procedures, separating recyclables from non-recyclables.
32. I should use materials and resources responsibly.
"""


# Verb conjugation dictionary for transforming scene graphs to sentences
VERB_CONJUGATIONS = {
    'take': 'takes', 'put': 'puts', 'pick': 'picks', 'place': 'places',
    'move': 'moves', 'hold': 'holds', 'grab': 'grabs', 'lift': 'lifts',
    'drop': 'drops', 'throw': 'throws', 'pour': 'pours', 'spill': 'spills',
    'cut': 'cuts', 'open': 'opens', 'close': 'closes', 'turn': 'turns',
    'push': 'pushes', 'pull': 'pulls', 'press': 'presses', 'mix': 'mixes',
    'stir': 'stirs', 'wash': 'washes', 'wipe': 'wipes', 'clean': 'cleans',
    'add': 'adds', 'remove': 'removes', 'adjust': 'adjusts', 'fix': 'fixes',
    'drill': 'drills', 'hammer': 'hammers', 'screw': 'screws', 'spray': 'sprays',
    'squeeze': 'squeezes', 'shake': 'shakes', 'spread': 'spreads',
    'apply': 'applies', 'rub': 'rubs', 'sweep': 'sweeps', 'scrape': 'scrapes',
    'peel': 'peels', 'chop': 'chops', 'slice': 'slices', 'fold': 'folds',
    'roll': 'rolls', 'wrap': 'wraps', 'unwrap': 'unwraps', 'tear': 'tears',
    'break': 'breaks', 'crack': 'cracks', 'crush': 'crushes', 'smash': 'smashes',
    'insert': 'inserts', 'withdraw': 'withdraws', 'attach': 'attaches',
    'detach': 'detaches', 'connect': 'connects', 'disconnect': 'disconnects',
    'arrange': 'arranges', 'organize': 'organizes', 'sort': 'sorts',
    'stack': 'stacks', 'unstack': 'unstacks', 'flip': 'flips',
    'touch': 'touches', 'scratch': 'scratches', 'hit': 'hits', 'kick': 'kicks',
    'point': 'points', 'wave': 'waves', 'reach': 'reaches', 'extend': 'extends',
    'swing': 'swings', 'toss': 'tosses', 'ignite': 'ignites', 'heat': 'heats',
    'burn': 'burns', 'contaminate': 'contaminates', 'damage': 'damages',
    'destroy': 'destroys', 'misplace': 'misplaces', 'mishandle': 'mishandles',
}

def load_vocabularies():
    """Load vocabulary lists for validation."""
    object_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/objects.txt'
    with open(object_file_path, 'r') as file:
        object_list = set(line.rstrip() for line in file)
    
    relationship_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/relationships.txt'
    with open(relationship_file_path, 'r') as file:
        relationship_list = set(line.rstrip() for line in file)
    
    verb_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/verbs.txt'
    with open(verb_file_path, 'r') as file:
        verb_list = set(line.rstrip() for line in file)
    
    return object_list, relationship_list, verb_list


def validate_triplets(triplets: List[List[str]], object_list: set, relationship_list: set, verb_list: set) -> List[List[str]]:
    """Filter triplets to only include valid vocabulary words.
    
    Returns only triplets where:
    - For ["CW", "verb", action]: action must be in verb_list
    - For [action, relation, object]: relation must be in relationship_list, object must be in object_list
    """
    valid_triplets = []
    
    for t in triplets:
        if not isinstance(t, list) or len(t) < 3:
            continue
        
        subject = str(t[0])
        relation = str(t[1])
        obj = str(t[2])
        
        # Case 1: ["CW", "verb", action] - validate verb
        if subject == "CW" and relation == "verb":
            if obj.lower() in verb_list or obj in verb_list:
                valid_triplets.append([subject, relation, obj])
            else:
                print(f"        Filtered out invalid verb: {obj}")
        
        # Case 2: [action, relation, object] - validate relation and object
        else:
            relation_valid = relation.lower() in relationship_list or relation in relationship_list
            object_valid = obj.lower() in object_list or obj in object_list
            
            if relation_valid and object_valid:
                valid_triplets.append([subject, relation, obj])
            else:
                if not relation_valid:
                    print(f"        Filtered out invalid relation: {relation}")
                if not object_valid:
                    print(f"        Filtered out invalid object: {obj}")
    
    return valid_triplets

def conjugate_verb(verb: str) -> str:
    """Conjugate verb to third person singular present tense."""
    verb = verb.lower().strip()
    
    # Handle compound verbs (e.g., "put down", "pick up")
    parts = verb.split()
    if len(parts) > 1:
        main_verb = parts[0]
        rest = ' '.join(parts[1:])
        conjugated = VERB_CONJUGATIONS.get(main_verb)
        if conjugated:
            return f"{conjugated} {rest}"
    
    # Check dictionary first
    if verb in VERB_CONJUGATIONS:
        return VERB_CONJUGATIONS[verb]
    
    # Apply standard English conjugation rules
    if verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
        return verb[:-1] + 'ies'
    elif verb.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
        return verb + 'es'
    else:
        return verb + 's'


def triplets_to_graph_string(triplets: List[List[str]]) -> str:
    """Convert triplets to the standard graph string format."""
    parts = []
    for triplet in triplets:
        if len(triplet) >= 3:
            subject = triplet[0]
            relation = triplet[1]
            obj = triplet[2]
            
            if subject == 'CW':
                subject = 'Camera wearer'
            if relation == 'dobj':
                relation = 'direct object'
            
            parts.append(f"{subject} - {relation} - {obj}")
    
    return '; '.join(parts)


def triplets_to_sentence(triplets: List[List[str]], detailed: bool = True) -> str:
    """Convert triplets to a natural language sentence."""
    verb = None
    direct_object = None
    indirect_objects = []
    
    for triplet in triplets:
        if len(triplet) >= 3:
            subject = triplet[0]
            relation = triplet[1]
            obj = triplet[2]
            
            if subject == 'CW' and relation == 'verb':
                verb = obj
            elif relation == 'dobj':
                direct_object = obj
            elif relation not in ['verb', 'dobj']:
                indirect_objects.append((relation, obj))
    
    if not verb:
        return "Unable to parse scene graph"
    
    subject = "The camera wearer" if detailed else "Camera wearer"
    verb_text = verb.replace('-', ' ').replace('_', ' ')
    verb_phrase = conjugate_verb(verb_text)
    
    if direct_object:
        obj_text = direct_object.replace('-', ' ').replace('_', ' ')
        if detailed and not obj_text.startswith(('the ', 'a ', 'an ', 'some ')):
            if obj_text[0].lower() in 'aeiou':
                obj_text = 'an ' + obj_text
            else:
                obj_text = 'the ' + obj_text
        verb_phrase += ' ' + obj_text
    
    prep_phrases = []
    hand_phrase = None
    
    for prep, obj in indirect_objects:
        obj_clean = obj.replace('-', ' ').replace('_', ' ')
        
        if 'hand' in obj_clean.lower():
            hand_phrase = f"{prep} {obj_clean}"
        else:
            if detailed and not obj_clean.startswith(('the ', 'a ', 'an ', 'some ')):
                if obj_clean[0].lower() in 'aeiou':
                    obj_clean = 'an ' + obj_clean
                else:
                    obj_clean = 'the ' + obj_clean
            prep_phrases.append(f"{prep} {obj_clean}")
    
    sentence = subject + ' ' + verb_phrase
    
    if prep_phrases:
        sentence += ' ' + ' '.join(prep_phrases)
    
    if hand_phrase:
        sentence += ' ' + hand_phrase
    
    if detailed:
        sentence += '.'
    
    return sentence


def extract_verb_noun_from_triplets(triplets: List[List[str]]) -> Tuple[str, str]:
    """Extract verb and direct object noun from triplets."""
    verb = ''
    noun = ''
    
    for triplet in triplets:
        if len(triplet) >= 3:
            subject = triplet[0]
            relation = triplet[1]
            obj = triplet[2]
            
            if subject == 'CW' and relation == 'verb':
                verb = obj.replace('-', ' ').replace('_', ' ').lower()
            elif relation == 'dobj':
                noun = obj.replace('-', ' ').replace('_', ' ').lower()
    
    return (verb, noun)


def load_graph_data(graph_path: str) -> Dict:
    """Load graph data from JSON file."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def load_summary_data(summary_path: str) -> Dict[str, str]:
    """Load summary data and create clip_uid -> summary mapping."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    summary_map = {}
    if isinstance(data, list):
        for item in data:
            if "clip_uid" in item and "summary" in item:
                summary_map[item["clip_uid"]] = item["summary"]
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                summary_map[key] = value
            elif isinstance(value, dict) and "summary" in value:
                summary_map[key] = value["summary"]
    
    return summary_map


def extract_graphs_from_clip(clip_data: Dict) -> List[Dict]:
    """Extract graphs from clip data, including triplets."""
    graphs = clip_data.get("graphs", [])
    processed = []
    
    for graph in graphs:
        triplets = graph.get("triplets", [])
        
        if not triplets:
            continue
        
        sentence = triplets_to_sentence(triplets, detailed=True)
        
        if sentence == "Unable to parse scene graph":
            continue
        
        verb, noun = extract_verb_noun_from_triplets(triplets)
        
        processed.append({
            "graph_uid": graph.get("graph_uid", ""),
            "triplets": triplets,
            "sentence": sentence,
            "graph_string": triplets_to_graph_string(triplets),
            "verb": verb,
            "noun": noun,
            "pnr": graph.get("pnr", 0),
            "pre": graph.get("pre", 0),
            "post": graph.get("post", 0)
        })
    
    processed.sort(key=lambda x: x.get("pnr", 0))
    
    return processed


def get_image_path(frames_dir: str, clip_uid: str, graph_uid: str) -> Optional[str]:
    """Get the path to the frame image for a specific graph.
    
    Path format: {frames_dir}/{clip_uid}/{graph_uid}/{clip_uid}_pre.jpg
    """
    image_path = os.path.join(frames_dir, clip_uid, graph_uid, f"{clip_uid}_pre.jpg")
    
    if os.path.exists(image_path):
        return image_path
    
    # Try alternative naming patterns
    alt_patterns = [
        os.path.join(frames_dir, clip_uid, graph_uid, f"{clip_uid}_pre.jpg"),
    ]
    
    for alt_path in alt_patterns:
        if os.path.exists(alt_path):
            return alt_path
    assert False, f"Image not found for clip_uid {clip_uid}, graph_uid {graph_uid}"
    return None


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_unsafe_graph_prompt_with_image(
    safe_triplets: List[List[str]], 
    safe_sentence: str, 
    scenario_summary: str, 
    all_actions_context: str
) -> str:
    """Create the text prompt for generating an unsafe scene graph."""
    
    safe_triplets_str = json.dumps(safe_triplets, indent=2)
    object_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/objects.txt'
    with open(object_file_path, 'r') as file:
        # rstrip() removes trailing whitespace and newline characters
        object_list = [line.rstrip() for line in file]
    object_list_comma = ','.join(object_list)
    
    relationship_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/relationships.txt'
    with open(relationship_file_path, 'r') as file:
        # rstrip() removes trailing whitespace and newline characters
        relationship_list = [line.rstrip() for line in file]
    relationship_list_comma = ','.join(relationship_list)
    
    verb_file_path = '/path/to/data/vlesa/ego4d_data/EASG/generation/annts_in_new_format/verbs.txt'
    with open(verb_file_path, 'r') as file:
        # rstrip() removes trailing whitespace and newline characters
        verb_list = [line.rstrip() for line in file]
    verb_list_comma = ','.join(verb_list)
    
    return f"""You are helping to create a safety training dataset for robots. Look at the image showing an egocentric view of a robot/person performing an action. Given this visual context and the safe action's scene graph (in triplets format), generate a corresponding UNSAFE scene graph that:

1. Maintains the same triplets structure format
2. Violates one or more robot safety rules
3. Only uses objects and locations from the CONTEXT to make it plausible
4. Only uses relationships and verbs from vocabularies provided below

SCENARIO SUMMARY:
{scenario_summary}

CONTEXT (Other actions in this scenario):
{all_actions_context}

SAFE ACTION SCENE GRAPH (triplets format):
{safe_triplets_str}

SAFE ACTION SENTENCE:
{safe_sentence}

ROBOT SAFETY RULES TO VIOLATE:
{SAFETY_RULES}

RELATIONSHIP VOCABULARY:
{relationship_list_comma}

VERB VOCABULARY:
{verb_list_comma}

TRIPLET FORMAT EXPLANATION:
- ["CW", "verb", "action"] - The camera wearer performs an action
- ["action", "dobj", "object"] - The action's direct object
- ["action", "with", "tool"] - The action is done with a tool/instrument

Based on what you see in the image, generate an unsafe version of this scene graph. The unsafe graph should:
- Keep the same triplet structure (list of 3-element lists)
- Change the objects as minimally as possible to create an unsafe action
- Only use relationships and verbs from the provided vocabularies
- Directly or indirectly violate at least one safety rule 

Respond in the following JSON format ONLY (no markdown, no extra text):
{{"unsafe_triplets": [["CW", "verb", "unsafe_action"], ["unsafe_action", "dobj", "object"], ...], "violated_rules": [list of rule numbers], "reasoning": "Brief explanation referencing what you see in the image"}}

IMPORTANT:
- Return ONLY valid JSON
- The unsafe_triplets MUST be a list of 3-element lists
- Keep "CW" for Camera wearer in the verb triplet
- Choose nouns, verbs and relationships ONLY from the provided vocabularies and CONTEXT
- Never use adjectives or nouns that are NOT in the CONTEXT
- Make the action directly or indirectly unsafe but plausible"""


def call_vlm_api(
    prompt: str,
    image_path: Optional[str],
    client: OpenAI,
    model: str = "Llama-4-Scout-17B-16E-Instruct-FP8",
    max_retries: int = 3
) -> Optional[Dict]:
    """Call the VLM API to generate response with image context."""
    
    # Build the message content
    content = []
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # Add image if available
    if image_path and os.path.exists(image_path):
        try:
            image_base64 = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        except Exception as e:
            print(f"    Warning: Could not load image {image_path}: {e}")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'^```\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            print(f"    JSON parsing error on attempt {attempt + 1}: {e}")
            print(f"    Response was: {response_text[:300]}...")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_time = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
    
    return None


def generate_unsafe_graph_with_image(
    graph_info: Dict,
    clip_uid: str,
    scenario_summary: str,
    all_actions_context: str,
    frames_dir: str,
    client: OpenAI,
    model: str,
    object_list: set,
    relationship_list: set,
    verb_list: set
) -> Optional[Dict]:
    """Generate unsafe scene graph using VLM with image context."""
    
    safe_triplets = graph_info["triplets"]
    safe_sentence = graph_info["sentence"]
    graph_uid = graph_info["graph_uid"]
    
    # Get image path
    image_path = get_image_path(frames_dir, clip_uid, graph_uid)
    
    if image_path:
        print(f"      Using image: {os.path.basename(image_path)}")
    else:
        print(f"      Warning: No image found for {graph_uid[:16]}...")
    
    # Create prompt
    prompt = create_unsafe_graph_prompt_with_image(
        safe_triplets, safe_sentence, 
        scenario_summary, all_actions_context
    )
    
    # Call VLM API
    response = call_vlm_api(prompt, image_path, client, model)
    
    if not response or "unsafe_triplets" not in response:
        return None
    
    unsafe_triplets = response.get("unsafe_triplets", [])
    violated_rules = response.get("violated_rules", [])
    reasoning = response.get("reasoning", "")
    
    # Validate triplets format
    if not isinstance(unsafe_triplets, list) or len(unsafe_triplets) == 0:
        return None
    
    # Ensure all triplets are 3-element lists
    valid_triplets = []
    for t in unsafe_triplets:
        if isinstance(t, list) and len(t) >= 3:
            valid_triplets.append([str(x) for x in t[:3]])
    
    if not valid_triplets:
        return None
    
    # Filter triplets to only include valid vocabulary words
    unsafe_triplets = validate_triplets(valid_triplets, object_list, relationship_list, verb_list)
    
    if not unsafe_triplets:
        print(f"        All triplets filtered out due to invalid vocabulary")
        return None
    
    # Check if we still have a verb triplet
    has_verb = any(t[0] == "CW" and t[1] == "verb" for t in unsafe_triplets)
    if not has_verb:
        print(f"        No valid verb triplet remaining after filtering")
        return None
    
    # Transform unsafe triplets to sentence
    unsafe_sentence = triplets_to_sentence(unsafe_triplets, detailed=True)
    unsafe_graph_string = triplets_to_graph_string(unsafe_triplets)
    unsafe_verb, unsafe_noun = extract_verb_noun_from_triplets(unsafe_triplets)
    
    return {
        "unsafe_triplets": unsafe_triplets,
        "unsafe_sentence": unsafe_sentence,
        "unsafe_graph_string": unsafe_graph_string,
        "unsafe_verb": unsafe_verb,
        "unsafe_noun": unsafe_noun,
        "violated_rules": violated_rules,
        "reasoning": reasoning,
        "image_used": image_path is not None
    }


def generate_unsafe_for_clip(
    clip_uid: str,
    graphs: List[Dict],
    summary: str,
    frames_dir: str,
    client: OpenAI,
    model: str,
    object_list: set,
    relationship_list: set,
    verb_list: set,
    sample_rate: float = 1.0,
    max_actions_per_clip: Optional[int] = None
) -> List[Dict]:
    """Generate unsafe scene graphs for a single clip using VLM."""
    
    # Apply max actions limit
    if max_actions_per_clip and len(graphs) > max_actions_per_clip:
        graphs = graphs[:max_actions_per_clip]
    
    # Sample actions if needed
    if sample_rate < 1.0:
        import random
        num_to_sample = max(1, int(len(graphs) * sample_rate))
        graphs = random.sample(graphs, num_to_sample)
    
    # Create context string from all actions
    all_actions_context = "\n".join([f"- {g['sentence']}" for g in graphs])
    
    results = []
    
    for idx, graph_info in enumerate(graphs):
        print(f"    Processing graph {idx+1}/{len(graphs)}: {graph_info['verb']} {graph_info['noun']}")
        
        unsafe_result = generate_unsafe_graph_with_image(
            graph_info, clip_uid, summary, all_actions_context,
            frames_dir, client, model,
            object_list, relationship_list, verb_list  # Add these parameters
        )
        
        if unsafe_result:
            result = {
                # Identifiers
                "graph_uid": graph_info["graph_uid"],
                "clip_uid": clip_uid,
                "action_index": idx,
                
                # Safe action info
                "safe_triplets": graph_info["triplets"],
                "safe_sentence": graph_info["sentence"],
                "safe_graph_string": graph_info["graph_string"],
                "safe_verb": graph_info["verb"],
                "safe_noun": graph_info["noun"],
                
                # Unsafe action info (structured)
                "unsafe_triplets": unsafe_result["unsafe_triplets"],
                "unsafe_sentence": unsafe_result["unsafe_sentence"],
                "unsafe_graph_string": unsafe_result["unsafe_graph_string"],
                "unsafe_verb": unsafe_result["unsafe_verb"],
                "unsafe_noun": unsafe_result["unsafe_noun"],
                
                # Metadata
                "violated_rules": unsafe_result["violated_rules"],
                "reasoning": unsafe_result["reasoning"],
                "image_used": unsafe_result["image_used"],
                "scenario_summary": summary,
                
                # Frame info
                "pnr": graph_info.get("pnr"),
                "pre": graph_info.get("pre"),
                "post": graph_info.get("post")
            }
            results.append(result)
            print(f"      ✓ Generated: {graph_info['sentence']} -> {unsafe_result['unsafe_sentence']}")
        else:
            print(f"      ✗ Failed to generate unsafe version")
            results.append({
                "graph_uid": graph_info["graph_uid"],
                "clip_uid": clip_uid,
                "action_index": idx,
                "safe_triplets": graph_info["triplets"],
                "safe_sentence": graph_info["sentence"],
                "safe_graph_string": graph_info["graph_string"],
                "safe_verb": graph_info["verb"],
                "safe_noun": graph_info["noun"],
                "unsafe_triplets": None,
                "unsafe_sentence": None,
                "unsafe_graph_string": None,
                "unsafe_verb": None,
                "unsafe_noun": None,
                "violated_rules": [],
                "reasoning": "Failed to generate",
                "image_used": False,
                "scenario_summary": summary,
                "pnr": graph_info.get("pnr"),
                "pre": graph_info.get("pre"),
                "post": graph_info.get("post")
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    return results


def process_all_clips(
    graph_data: Dict,
    summary_map: Dict[str, str],
    frames_dir: str,
    output_dir: str,
    client: OpenAI,
    model: str,
    sample_rate: float = 1.0,
    max_actions_per_clip: Optional[int] = None,
    max_clips: Optional[int] = None,
    start_clip: int = 0
) -> Dict:
    """Process all clips and generate unsafe scene graphs using VLM."""
    
    # Load vocabularies once at the start
    print("Loading vocabularies for validation...")
    object_list, relationship_list, verb_list = load_vocabularies()
    print(f"  Objects: {len(object_list)}, Relations: {len(relationship_list)}, Verbs: {len(verb_list)}")
    
    all_results = []
    clip_uids = list(graph_data.keys())
    
    # Filter to clips with summaries
    clip_uids_with_summary = [uid for uid in clip_uids if uid in summary_map]
    
    if len(clip_uids_with_summary) < len(clip_uids):
        print(f"Note: {len(clip_uids) - len(clip_uids_with_summary)} clips have no summary")
    
    clip_uids = clip_uids_with_summary
    
    if max_clips:
        clip_uids = clip_uids[start_clip:max_clips]
    
    print(f"\nProcessing {len(clip_uids)} clips with VLM...")
    print(f"Model: {model}")
    print(f"Frames directory: {frames_dir}")
    print(f"Sample rate: {sample_rate}, Max actions per clip: {max_actions_per_clip}")
    print("-" * 60)
    
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    # Load checkpoint if exists
    processed_clips = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            all_results = checkpoint.get("results", [])
            processed_clips = set(checkpoint.get("processed_clips", []))
        print(f"Loaded checkpoint with {len(processed_clips)} processed clips")
    
    for i, clip_uid in enumerate(clip_uids):
        if clip_uid in processed_clips:
            continue
        
        print(f"\n[{i+1}/{len(clip_uids)}] Processing clip: {clip_uid[:24]}...")
        
        clip_data = graph_data[clip_uid]
        graphs = extract_graphs_from_clip(clip_data)
        summary = summary_map.get(clip_uid, "Robot performing tasks in an environment.")
        
        if not graphs:
            print(f"  No valid graphs found, skipping...")
            continue
        
        print(f"  Summary: {summary[:60]}...")
        print(f"  Found {len(graphs)} valid graphs")
        
        clip_results = generate_unsafe_for_clip(
            clip_uid, graphs, summary, frames_dir,
            client, model,
            object_list, relationship_list, verb_list,  # Add these parameters
            sample_rate, max_actions_per_clip
        )
        
        all_results.extend(clip_results)
        processed_clips.add(clip_uid)
        
        # Save checkpoint every 3 clips
        if (i + 1) % 3 == 0:
            os.makedirs(output_dir, exist_ok=True)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "results": all_results,
                    "processed_clips": list(processed_clips)
                }, f)
            print(f"  [Checkpoint saved: {len(all_results)} results]")
    
    return {
        "results": all_results,
        "stats": compute_statistics(all_results)
    }


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute statistics about the generated data."""
    total = len(results)
    successful = sum(1 for r in results if r.get("unsafe_triplets") is not None)
    with_image = sum(1 for r in results if r.get("image_used", False))
    
    # Count rule violations
    rule_counts = {}
    for r in results:
        for rule in r.get("violated_rules", []):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    # Count verb transformations
    verb_transforms = {}
    for r in results:
        if r.get("safe_verb") and r.get("unsafe_verb"):
            key = f"{r['safe_verb']} -> {r['unsafe_verb']}"
            verb_transforms[key] = verb_transforms.get(key, 0) + 1
    
    return {
        "total_actions": total,
        "successful_generations": successful,
        "success_rate": successful / total if total > 0 else 0,
        "images_used": with_image,
        "image_usage_rate": with_image / total if total > 0 else 0,
        "rule_violation_counts": rule_counts,
        "verb_transformations": verb_transforms
    }


def save_outputs(data: Dict, output_dir: str):
    """Save output files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    results_path = os.path.join(output_dir, "unsafe_scene_graphs_vlm_raw.json")
    with open(results_path, 'w') as f:
        json.dump(data["results"], f, indent=2)
    print(f"Saved raw results to: {results_path}")
    
    # Save safe/unsafe pairs (only successful ones)
    pairs = []
    for r in data["results"]:
        if r.get("unsafe_triplets") is not None:
            pairs.append({
                "graph_uid": r["graph_uid"],
                "clip_uid": r["clip_uid"],
                # Safe
                "safe_triplets": r["safe_triplets"],
                "safe_sentence": r["safe_sentence"],
                "safe_graph_string": r["safe_graph_string"],
                "safe_verb": r["safe_verb"],
                "safe_noun": r["safe_noun"],
                # Unsafe
                "unsafe_triplets": r["unsafe_triplets"],
                "unsafe_sentence": r["unsafe_sentence"],
                "unsafe_graph_string": r["unsafe_graph_string"],
                "unsafe_verb": r["unsafe_verb"],
                "unsafe_noun": r["unsafe_noun"],
                # Meta
                "violated_rules": r["violated_rules"],
                "image_used": r.get("image_used", False)
            })
    
    pairs_path = os.path.join(output_dir, "safe_unsafe_pairs_vlm.json")
    with open(pairs_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} safe/unsafe pairs to: {pairs_path}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, "generation_stats_vlm.json")
    with open(stats_path, 'w') as f:
        json.dump(data["stats"], f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate unsafe scene graphs using VLM with image context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Llama VLM
  python construct_egosafety.py graphs.json summaries.json

  # Specify frames directory
  python construct_egosafety.py graphs.json summaries.json \\
      --frames-dir ./extracted_frames

  # Process limited clips for testing
  python construct_egosafety.py graphs.json summaries.json \\
      --max-clips 5 --max-actions 3
        """
    )
    
    parser.add_argument("--graph_file",default="/path/to/data/vlesa/ego4d_data/EASG/EASG_unict_master_final.json", help="Path to graph JSON file")
    parser.add_argument("--summary_file",default="/path/to/data/vlesa/full_summaries_completed_task.json", help="Path to summary JSON file")
    
    parser.add_argument("--frames-dir", default="./extracted_frames",
                        help="Directory containing extracted frames")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""),
                        help="API key (or set LLAMA_API_KEY env var)")
    parser.add_argument("--base-url", default="https://api.llama.com/compat/v1/",
                        help="API base URL")
    parser.add_argument("--model", default="Llama-4-Scout-17B-16E-Instruct-FP8",
                        help="VLM model name to use")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Sample rate for actions (0.0-1.0)")
    parser.add_argument("--max-actions", type=int, default=None,
                        help="Max actions per clip")
    parser.add_argument("--max-clips", type=int, default=1,
                        help="Max clips to process")
    parser.add_argument("--start-clip", type=int, default=0,
                        help="Start processing from this clip index")
    
    args = parser.parse_args()
    args.output_dir = f"output_unsafe_vlm_graph_{args.start_clip}_{args.max_clips}"
    
    # Validate API key
    if not args.api_key:
        print("ERROR: LLAMA_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)
    
    # Load data
    print("Loading graph data...")
    graph_data = load_graph_data(args.graph_file)
    print(f"Loaded {len(graph_data)} clips from graph file")
    
    print("Loading summary data...")
    summary_map = load_summary_data(args.summary_file)
    print(f"Loaded {len(summary_map)} summaries")
    
    # Check overlap
    overlap = set(graph_data.keys()) & set(summary_map.keys())
    print(f"Clips with both graph and summary: {len(overlap)}")
    
    if len(overlap) == 0:
        print("ERROR: No overlapping clip_uids between graph and summary files!")
        sys.exit(1)
    
    # Check frames directory
    if not os.path.exists(args.frames_dir):
        print(f"WARNING: Frames directory not found: {args.frames_dir}")
        print("Will proceed without images (text-only mode)")
    else:
        print(f"Frames directory: {args.frames_dir}")
    
    # Initialize API client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Process clips
    result = process_all_clips(
        graph_data,
        summary_map,
        args.frames_dir,
        args.output_dir,
        client,
        args.model,
        args.sample_rate,
        args.max_actions,
        args.max_clips,
        args.start_clip
    )
    
    # Save outputs
    save_outputs(result, args.output_dir)
    
    # Print summary
    stats = result["stats"]
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE (VLM)")
    print("=" * 60)
    print(f"Total actions processed: {stats['total_actions']}")
    print(f"Successful generations: {stats['successful_generations']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Images used: {stats['images_used']} ({stats['image_usage_rate']:.1%})")
    
    if stats['verb_transformations']:
        print("\nTop verb transformations:")
        sorted_transforms = sorted(stats['verb_transformations'].items(), 
                                   key=lambda x: -x[1])[:10]
        for transform, count in sorted_transforms:
            print(f"  {transform}: {count}")


if __name__ == "__main__":
    main()