import base64
import json
import os
import time
import requests
import numpy as np
from typing import Dict, List, Any, Tuple
import argparse
import sys

# Import the game environment
from bomberman_gym import BombermanEnv, BombermanAction
import re

class AIPlayerController:
    """AI Player Controller, controls the game character using an API"""
    
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base
        self.model_name = model_name
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
        
        # Initialize system prompts
        self.system_prompts = {
            0: self._create_system_prompt(0, RED=255, BLUE=0, GREEN=0),    # Red player
            1: self._create_system_prompt(1, RED=0, BLUE=255, GREEN=0),    # Blue player
            2: self._create_system_prompt(2, RED=0, BLUE=0, GREEN=255),    # Green player
            3: self._create_system_prompt(3, RED=255, BLUE=255, GREEN=0)   # Yellow player
        }
        
        # Store the game state of the previous step to generate a summary
        self.previous_states = {i: None for i in range(4)}
        self.game_history = {i: [] for i in range(4)}
    
    def _create_system_prompt(self, player_id: int, **kwargs) -> str:
        """Creates the system prompt"""
        color_name = ["Red", "Blue", "Green", "Yellow"][player_id]
        return f"""You are an AI player playing Bomberman, acting as the {color_name} character (Player {player_id+1}).
You need to make intelligent decisions based on the current game state information, game screen images, and sound events.

Game Rules:
1. You can move on the map or place bombs.
2. Bombs create cross-shaped explosions that can destroy soft walls and hit players.
3. Soft walls (brown blocks), when destroyed, may drop power-ups: increase firepower (increase explosion range), increase the number of bombs, or increase movement speed.
4. Players hit by flames will be killed. Your goal is to defeat other players and survive as long as possible.
5. The last surviving player wins.

Map Elements:
- Empty Space: Can move freely.
- Soft Wall (brown): Can be destroyed by bombs.
- Hard Wall (gray): Cannot be destroyed or passed through.
- Bomb: Explodes after being placed, creating cross-shaped flames.
- Flame: Harms players and destroys soft walls.
- Power-up: Enhances player abilities.

After analyzing the image, sound, and game state information, make the best decision:
1. Move to a safe location to avoid being hit by bombs.
2. Strategically place bombs to destroy soft walls or defeat opponents.
3. Collect valuable power-ups to improve your abilities.
4. Predict opponents' actions and react accordingly.

Please return your decision in JSON format:
For movement: {{"action_type": 0, "target_x": <target x-coordinate>, "target_y": <target y-coordinate>}}, for example: {{"action_type": 0, "target_x": 1, "target_y": 2}} But please ensure the target coordinates are within the map boundaries and do not exceed the maximum movement distance.
For placing a bomb: {{"action_type": 1, "target_x": 0, "target_y": 0}}

Ensure the returned format strictly adheres to the requirements. Only return a valid JSON object, without any additional explanatory text.
"""
    
    def get_decision(self, player_id: int, obs: Dict) -> Dict:
        """Get the AI's decision"""
        try:
            # Get basic information
            game_image = obs.get('image', '')
            audio_data = obs.get('audio', '')
            
            # Parse audio events
            audio_events = self._parse_audio_data(audio_data)
            
            # Generate user prompt content
            user_content = self._create_user_content(player_id, obs, audio_events)
            
            # Build request messages
            messages = [
                {"role": "system", "content": self.system_prompts[player_id]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content}
                    ]
                }
            ]
            
            # Add image data
            if game_image:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{game_image}"},
                })
            
            # Add audio data - optimized version
            if audio_events:
                # Add overall audio event description
                if len(audio_events) > 0:
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"There are a total of {len(audio_events)} audio events this turn:"
                    })
                
                # Iterate through all audio events
                for event_index, event in enumerate(audio_events):
                    player_name = f"Player {event['player_id'] + 1}"
                    event_type = event['event_type']
                    
                    # Build a friendly English description based on the event type
                    event_description = ""
                    if event_type == 'player_walk':
                        event_description = f"{player_name} walking footsteps"
                    elif event_type == 'bomb_place':
                        event_description = f"{player_name} placing a bomb"
                    elif event_type == 'bomb_explode':
                        event_description = f"{player_name}'s bomb exploding"
                    else:
                        event_description = f"{player_name}'s {event_type} sound"
                    
                    # Add audio description text
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"Audio {event_index+1}: {event_description}"
                    })
                    
                    # Add detailed information
                    if 'description' in event:
                        messages[1]["content"].append({
                            "type": "text", 
                            "text": f"Details: {event['description']}"
                        })
                    
                    # Add actual audio data
                    audio_b64 = event.get('audio_base64', '')
                    if audio_b64:
                        # Determine audio format
                        audio_format = "wav"  # default format
                        if "footstep" in event_type or "walk" in event_type:
                            audio_format = "wav"
                        elif "explosion" in event_type or "explode" in event_type:
                            audio_format = "wav"
                        elif "click" in event_type or "place" in event_type:
                            audio_format = "wav"
                        
                        messages[1]["content"].append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        })
            
            # Build the complete request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # Send API request
            print(f"Requesting decision for Player {player_id+1}...")
            r = self.session.post(f"{self.api_base}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            
            response_data = r.json()
            action_text = response_data['choices'][0]['message']['content']
            
            # Parse the returned JSON
            try:
                # Extract JSON part from the text
                action_text = action_text.strip()
                if action_text.startswith("```json"):
                    action_text = action_text.split("```json")[1].split("```").strip()
                elif action_text.startswith("```"):
                    action_text = action_text.split("```").split("```")[0].strip()
                
                action = json.loads(action_text)
                
                # Record to history
                action_desc = f"Move to ({action['target_x']},{action['target_y']})" if action['action_type'] == 0 else "Place bomb"
                self.game_history[player_id].append({
                    'step': obs['step'],
                    'action': action_desc
                })
                
                print(f"Player {player_id+1} decided: {action_desc}")
                return action
            except json.JSONDecodeError:
                print(f"Failed to parse JSON, returning default action. Original response: {action_text}")
                # Return default action
                return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}
                
        except Exception as e:
            print(f"Error in Player {player_id+1}'s decision: {e}")
            # When an error occurs, return a safe default action
            return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}

    # Add helper methods
    def _parse_audio_data(self, audio_data: str) -> List[Dict]:
        """Parse audio data"""
        try:
            if not audio_data:
                return []
            return json.loads(audio_data)
        except Exception as e:
            print(f"Error parsing audio data: {e}")
            return []
    
    def _create_user_content(self, player_id: int, obs: Dict, game_events: List[Dict]) -> str:
        """Create user prompt content"""
        # Get player information
        player_info = obs['state']['players'][player_id]
        position_x = player_info['position_x']
        position_y = player_info['position_y']
        
        # Create game event description
        game_events_description = self._format_game_events(game_events)
        
        # Create state change description (compared to the previous step)
        state_changes = self._create_state_changes_description(player_id, obs)
        
        # Other players' positions
        other_players = []
        for pid, p_info in obs['state']['players'].items():
            if int(pid) != player_id and p_info['alive'] == 1:
                other_players.append(f"Player {int(pid)+1}: ({p_info['position_x']}, {p_info['position_y']})")
        
        # Bomb information collection
        bombs = []
        danger_zones = set()  # Store coordinates of all danger zones
        for i in range(obs['state']['bombs']['count']):
            x = obs['state']['bombs']['positions_x'][i]
            y = obs['state']['bombs']['positions_y'][i]
            timer = obs['state']['bombs']['countdown'][i]
            owner = obs['state']['bombs']['owner'][i]
            fire = obs['state']['bombs']['fire_power'][i]
            bombs.append(f"Bomb: Position({x},{y}), Countdown {timer} steps, Owner Player {owner+1}, Firepower {fire}")
            
            # Calculate the danger zone for this bomb
            # Center point
            danger_zones.add((x, y))
            # Four directions
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                for dist in range(1, fire + 1):
                    danger_x = x + dx * dist
                    danger_y = y + dy * dist
                    # Check boundaries
                    if 0 <= danger_x < 13 and 0 <= danger_y < 11:
                        danger_zones.add((danger_x, danger_y))
        
        # Check if the player is in a danger zone
        player_in_danger = (position_x, position_y) in danger_zones
        
        # Tally recent bombs
        bombs = []
        for i in range(obs['state']['bombs']['count']):
            x = obs['state']['bombs']['positions_x'][i]
            y = obs['state']['bombs']['positions_y'][i]
            timer = obs['state']['bombs']['countdown'][i]
            owner = obs['state']['bombs']['owner'][i]
            fire = obs['state']['bombs']['fire_power'][i]
            bombs.append(f"Bomb: Position({x},{y}), Countdown {timer} steps, Owner Player {owner+1}, Firepower {fire}")
        
        # Create game history summary
        history_summary = ""
        if len(self.game_history[player_id]) > 0:
            recent_history = self.game_history[player_id][-5:]  # last 5 records
            history_summary = "Recent Action History:\n" + "\n".join([f"- Turn {h['step']}: {h['action']}" for h in recent_history])
        
        # Calculate the current player's maximum movement distance
        max_move_distance = 5 + (player_info['speed'] - 1)  # base distance 5 + speed bonus
        
        # Build the complete user prompt
        return f"""Current Game State Analysis - Turn {obs['step']}:

You are Player {player_id+1}, {"Alive" if player_info['alive'] == 1 else "Dead"}
Your Position: ({position_x}, {position_y})
Your Attributes:
- Firepower: {player_info['fire_power']} (bomb explosion range)
- Bomb Count: {player_info['bomb_count']} (number of bombs you can place simultaneously)
- Active Bombs Placed: {player_info['active_bombs']}
- Movement Speed: {player_info['speed']}
- Trapped Status: {"Yes" if player_info['trapped'] == 1 else "No"}

{"⚠️ Warning: You are currently in a bomb's blast range! Evacuate immediately!" if player_in_danger else ""}

Movement Ability Limits:
- Your maximum movement distance is {max_move_distance} squares (Manhattan distance)
- This is the base distance (5 squares) plus the bonus from the speed attribute (speed value - 1)
- You cannot pass through walls or bombs - A placed bomb becomes an obstacle you cannot pass through
- If the target position is out of range, you will move to the furthest reachable point

Other Players' Positions:
{chr(10).join(other_players) if other_players else "No other surviving players"}

Danger Zone Warning:
{chr(10).join([f"⚠️ Bomb at ({x},{y}), will explode in {timer} steps, firepower range {fire} squares, affecting horizontal area from ({x-fire},{y}) to ({x+fire},{y}) and vertical area from ({x},{y-fire}) to ({x},{y+fire})!" 
        for i, (x, y, timer, owner, fire) in enumerate([(obs['state']['bombs']['positions_x'][i], 
                                 obs['state']['bombs']['positions_y'][i], 
                                 obs['state']['bombs']['countdown'][i],
                                 obs['state']['bombs']['owner'][i],
                                 obs['state']['bombs']['fire_power'][i]) 
                               for i in range(obs['state']['bombs']['count'])])]) if bombs else "No bomb threats on the field currently"}

Important Reminder: A placed bomb becomes an impassable obstacle! Consider this when planning your path.

{state_changes}

{game_events_description}

{history_summary}

Please review the attached game screen image and sound events, analyze the current situation, and decide whether to move to a safe location, place a bomb, or collect a power-up.
Prioritize safety! Stay away from areas of impending explosions, especially from bombs with short countdowns.
Return a JSON action in the correct format, for example {{"action_type": 0, "target_x": 5, "target_y": 3}} to move to position (5,3).
"""
    
    def _format_game_events(self, game_events: List[Dict]) -> str:
        """Format game event descriptions"""
        if not game_events:
            return "No special game events this turn."
        
        events_str = "Game Events This Turn:"
        for event in game_events:
            event_type = event['event_type']
            player_name = f"Player {event['player_id']+1}"
            
            if event_type == 'player_walk':
                from_pos = event['params']['from_pos']
                to_pos = event['params']['to_pos']
                events_str += f"\n- {player_name} moved from ({from_pos[0]},{from_pos[1]}) to ({to_pos[0]},{to_pos[1]})"
            
            elif event_type == 'bomb_place':
                pos = event['params']['pos']
                fire = event['params']['fire']
                events_str += f"\n- {player_name} placed a bomb with firepower {fire} at ({pos[0]},{pos[1]})"
            
            elif event_type == 'bomb_explode':
                pos = event['params']['pos']
                fire = event['params']['fire']
                affected = len(event['params']['affected_positions'])
                events_str += f"\n- {player_name}'s bomb exploded at ({pos[0]},{pos[1]}) with firepower {fire}, affecting {affected} positions"
        
        return events_str
    
    def _create_state_changes_description(self, player_id: int, obs: Dict) -> str:
        """Create a description of state changes compared to the previous step"""
        if not self.previous_states[player_id]:
            self.previous_states[player_id] = obs
            return "This is the first turn of the game."
        
        prev = self.previous_states[player_id]
        changes = []
        
        # Check for player state changes
        curr_player = obs['state']['players'][player_id]
        prev_player = prev['state']['players'][player_id]
        
        if curr_player['fire_power'] > prev_player['fire_power']:
            changes.append(f"Your firepower increased: {prev_player['fire_power']} → {curr_player['fire_power']}")
        
        if curr_player['bomb_count'] > prev_player['bomb_count']:
            changes.append(f"Your bomb count increased: {prev_player['bomb_count']} → {curr_player['bomb_count']}")
        
        if curr_player['speed'] > prev_player['speed']:
            changes.append(f"Your movement speed increased: {prev_player['speed']} → {curr_player['speed']}")
        
        if curr_player['trapped'] > prev_player['trapped']:
            changes.append("You are trapped by bomb flames! You need a teammate's help or you will die soon.")
        
        if curr_player['trapped'] < prev_player['trapped'] and prev_player['trapped'] == 1:
            changes.append("You have successfully escaped from being trapped!")
        
        # Check for changes in the number of bombs
        prev_bombs = prev['state']['bombs']['count']
        curr_bombs = obs['state']['bombs']['count']
        if curr_bombs > prev_bombs:
            changes.append(f"Number of bombs on the field increased: {prev_bombs} → {curr_bombs}")
        elif curr_bombs < prev_bombs:
            changes.append(f"A bomb exploded on the field, number of bombs decreased: {prev_bombs} → {curr_bombs}")
        
        # Check other players' survival status
        for pid, p_info in prev['state']['players'].items():
            pid = int(pid)
            if pid != player_id:
                prev_alive = p_info['alive']
                curr_alive = obs['state']['players'][pid]['alive']
                if prev_alive == 1 and curr_alive == 0:
                    changes.append(f"Player {pid+1} has died!")
        
        # Update state
        self.previous_states[player_id] = obs
        
        if not changes:
            return "No significant state changes compared to the previous step."
        return "State Changes:\n- " + "\n- ".join(changes)
    
    def _parse_audio_data(self, audio_data: str) -> List[Dict]:
        """Parse audio data"""
        try:
            if not audio_data:
                return []
            return json.loads(audio_data)
        except Exception as e:
            print(f"Error parsing audio data: {e}")
            return []
    
    def get_decision(self, player_id: int, obs: Dict) -> Dict:
        """Get the AI's decision"""
        try:
            # Get basic information
            game_image = obs.get('image', '')
            audio_data = obs.get('audio', '')
            
            # Parse audio events
            audio_events = self._parse_audio_data(audio_data)
            
            # Generate user prompt content
            user_content = self._create_user_content(player_id, obs, audio_events)
            
            # Build request messages
            messages = [
                {"role": "system", "content": self.system_prompts[player_id]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content}
                    ]
                }
            ]
            
            # Add image data
            if game_image:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{game_image}"},
                })
            
            # Add audio data - optimized version
            if audio_events:
                # Add overall audio event description
                if len(audio_events) > 0:
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"There are a total of {len(audio_events)} audio events this turn:"
                    })
                
                # Iterate through all audio events
                for event_index, event in enumerate(audio_events):
                    player_name = f"Player {event['player_id'] + 1}"
                    event_type = event['event_type']
                    
                    # Build a friendly English description based on the event type
                    event_description = ""
                    if event_type == 'player_walk':
                        event_description = f"{player_name} walking footsteps"
                    elif event_type == 'bomb_place':
                        event_description = f"{player_name} placing a bomb"
                    elif event_type == 'bomb_explode':
                        event_description = f"{player_name}'s bomb exploding"
                    else:
                        event_description = f"{player_name}'s {event_type} sound"
                    
                    # Add audio description text
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"Audio {event_index+1}: {event_description}"
                    })
                    
                    # Add detailed information
                    if 'description' in event:
                        messages[1]["content"].append({
                            "type": "text", 
                            "text": f"Details: {event['description']}"
                        })
                    
                    # Add actual audio data
                    audio_b64 = event.get('audio_base64', '')
                    if audio_b64:
                        # Determine audio format
                        audio_format = "wav"  # default format
                        if "footstep" in event_type or "walk" in event_type:
                            audio_format = "wav"
                        elif "explosion" in event_type or "explode" in event_type:
                            audio_format = "wav"
                        elif "click" in event_type or "place" in event_type:
                            audio_format = "wav"
                        
                        messages[1]["content"].append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        })
            
            # Build the complete request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # Send API request
            print(f"Requesting decision for Player {player_id+1}...")
            r = self.session.post(f"{self.api_base}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            
            response_data = r.json()
            action_text = response_data['choices'][0]['message']['content']
            
            # Parse the returned JSON
            try:
                # Extract JSON part from the text
                action_text = action_text.strip()
                action_json = None
                
                # Method 1: Check if it is in a code block format
                if "```json" in action_text:
                    action_json = action_text.split("```json")[1].split("```").strip()
                elif "```" in action_text:
                    action_json = action_text.split("```").split("```")[0].strip()
                
                # Method 2: Directly search for the JSON object pattern {...}
                if not action_json:
                    json_pattern = re.search(r'\{(?:[^{}]|"[^"]*")*\}', action_text)
                    if json_pattern:
                        action_json = json_pattern.group(0)
                
                # Parse JSON
                if action_json:
                    action = json.loads(action_json)
                else:
                    # If formatted JSON is not found, try parsing the entire response directly
                    action = json.loads(action_text)
                
                # Validate key fields
                required_fields = ["action_type", "target_x", "target_y"]
                if not all(field in action for field in required_fields):
                    raise ValueError(f"Missing required action fields: {required_fields}")
                
                # Record to history
                action_desc = f"Move to ({action['target_x']},{action['target_y']})" if action['action_type'] == 0 else "Place bomb"
                self.game_history[player_id].append({
                    'step': obs['step'],
                    'action': action_desc
                })
                
                print(f"Player {player_id+1} decided: {action_desc}")
                return action
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON: {e}, returning default action. Original response: {action_text[:100]}...")
                # Return default action
                return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}
        except Exception as e:
            print(f"Error in Player {player_id+1}'s decision: {e}")
            # When an error occurs, return a safe default action
            return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}

def run_ai_game(api_base, api_key, model_name, episodes=3, steps_per_episode=300, delay=0.1):
    """Run the AI-controlled game"""
    # Initialize the game environment
    env = BombermanEnv(render_mode='human')
    
    # Create the AI controller
    controller = AIPlayerController(api_base, api_key, model_name)
    
    try:
        for episode in range(episodes):
            print(f"\n====== Game {episode+1} ======")
            obs, info = env.reset()
            
            for step in range(steps_per_episode):
                # Get decisions for all active players
                actions = {}
                for player_id in range(env.num_players):
                    # Check if the player is alive
                    if obs['state']['players'][player_id]['alive'] == 0:
                        continue
                    
                    # Get AI decision
                    action = controller.get_decision(player_id, obs)
                    actions[player_id] = action
                
                # Execute actions
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Add a slight delay in the GUI for observation
                time.sleep(delay)
                
                # Check if the game is over
                if terminated:
                    # Find the winner
                    winner_id = None
                    for pid, p_info in obs['state']['players'].items():
                        if p_info['alive'] == 1:
                            winner_id = int(pid)
                            break
                    
                    if winner_id is not None:
                        print(f"Game Over! Player {winner_id+1} wins!")
                    else:
                        print("Game Over! It's a draw!")
                    break
                elif truncated:
                    print(f"Maximum steps {steps_per_episode} reached!")
                    break
    
    except KeyboardInterrupt:
        print("User interrupted, ending game")
    except Exception as e:
        print(f"An error occurred during the game: {e}")
    finally:
        # Ensure the environment is closed properly
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-controlled Bomberman Game")
    parser.add_argument("--api-base", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, required=True, help="API key")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--episodes", type=int, default=3, help="Number of game episodes")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay per step (seconds)")
    
    args = parser.parse_args()
    
    run_ai_game(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        delay=args.delay
    )