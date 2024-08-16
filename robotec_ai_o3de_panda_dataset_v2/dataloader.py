import numpy as np
import os
from PIL import Image
from typing import TypedDict, List, Dict
import json
import tqdm
import random

DATASET_PATH = "/robo-srv-004-storage-001/home/mkotynia/openvla_dataset/dataset6_500"
OUTPUT_PATH = "data"
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_PATH, "train")
VAL_SPLIT_PATH = os.path.join(OUTPUT_PATH, "val")
os.makedirs(TRAIN_SPLIT_PATH, exist_ok=True)
os.makedirs(VAL_SPLIT_PATH, exist_ok=True)

N_TRAIN_EPISODES = 500
N_VAL_EPISODES = 0

class Action(TypedDict):
    action: np.ndarray[np.float32]

class State(TypedDict):
    state: np.ndarray[np.float32]

def create_episodes(path: str, output_path: str, train_episodes: int, val_episodes: int, shuffle: bool = True):
    tasks = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    episodes = []
    
    for j, task in enumerate(tasks):
        episodes_path = os.path.join(task, "episodes")
        episodes_paths = [os.path.join(episodes_path, d) for d in os.listdir(episodes_path) if os.path.isdir(os.path.join(episodes_path, d))]
        language_instruction_path = os.path.join(task, "language_task.txt")
        language_instruction = read_string_from_file(language_instruction_path)
        
        for i, episode_path in tqdm.tqdm(enumerate(episodes_paths)):
            steps = [os.path.join(episode_path, d) for d in os.listdir(episode_path) if os.path.isdir(os.path.join(episode_path, d))]
            episode = []
            
            for step in steps:
                # image 1
                image1_path = os.path.join(step, "obs1.png")
                image1 = np.asarray(Image.open(image1_path), dtype=np.uint8)
                # image 2
                image2_path = os.path.join(step, "obs2.png")
                image2 = np.asarray(Image.open(image2_path), dtype=np.uint8)

                # image 3
                image3_path = os.path.join(step, "obs3.png")
                image3 = np.asarray(Image.open(image3_path), dtype=np.uint8)
                # image 4
                image4_path = os.path.join(step, "obs4.png")
                image4 = np.asarray(Image.open(image4_path), dtype=np.uint8)

                state = load_state(os.path.join(step, "state.json"))
                action = load_action(os.path.join(step, "action.json"))
                
                episode.append({
                    "image1": image1,
                    "image2": image2,
                    "image3": image3,
                    "image4": image4,
                    "state": np.asarray(state['state'], dtype=np.float32),
                    "action": np.asarray(action['action'], dtype=np.float32),
                    "language_instruction": language_instruction,
                })
                
            episodes.append((j, i, episode))
    
    if shuffle:
        random.shuffle(episodes)
    
    train_episodes_data = episodes[:train_episodes]
    val_episodes_data = episodes[train_episodes:train_episodes + val_episodes]
    
    save_episodes(train_episodes_data, TRAIN_SPLIT_PATH, "train")
    save_episodes(val_episodes_data, VAL_SPLIT_PATH, "val")

def save_episodes(episodes: List[Dict], output_path: str, split: str):
    for task_num, episode_num, episode in episodes:
        # np.save(os.path.join(output_path, f"{split}_task_{task_num}_episode_{episode_num}.npy"), episode)
        np.save(os.path.join(output_path, f"episode_{episode_num}.npy"), episode)

def read_string_from_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_action(path: str) -> Action:
    try:
        with open(path, "r") as f:
            action_data = json.load(f)
        return Action(action=np.asarray(action_data['action'], dtype=np.float32))
    except Exception as e:
        print(f"An error occurred while loading action: {e}")
        return None

def load_state(path: str) -> State:
    try:
        with open(path, "r") as f:
            state_data = json.load(f)
        return State(state=np.asarray(state_data['state'], dtype=np.float32))
    except Exception as e:
        print(f"An error occurred while loading state: {e}")
        return None

if __name__ == "__main__":
    create_episodes(DATASET_PATH, OUTPUT_PATH, N_TRAIN_EPISODES, N_VAL_EPISODES)
