import numpy as np
import pygame
import io
import base64
import json
import os
import tempfile
import time
import requests
from PIL import Image
import datetime
import pathlib
import re
from moviepy.editor import VideoFileClip
import cv2


API_BASE = ""
API_KEY = ""
MODEL_CHAT = "gemini-2.5-pro"


# Import your game environment
from rhythm_memory_gym_env import RhythmMemoryEnv  # Please replace with the actual import path


class ModelRhythmMemoryRunner:
    def __init__(self, difficulty=1, max_episodes=50):
        # Initialize Pygame
        pygame.init()
        pygame.mixer.init()
        
        # Store game settings
        self.difficulty = difficulty    
        self.max_episodes = max_episodes
        # Create game environment
        self.env = RhythmMemoryEnv(difficulty)  
        # Initialize HTTP session
        self.session = requests.Session()    
        
        # Use a fixed path under the project directory instead of the system temp directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join(base_dir, "ai_data", f"rhythm_memory_ai_{timestamp}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create directory for saving model input data
        self.model_input_dir = os.path.join(self.temp_dir, "model_inputs")
        os.makedirs(self.model_input_dir, exist_ok=True)
        # Counter for naming saved files
        self.input_counter = 0  

        # Game statistics
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_score": 0,
            "sequence_analysis_errors": 0,
            "click_prediction_errors": 0,
            "unparseable_sequences": 0,
            # Store detailed data for each episode
            "episodes_data": []  
        }

        # Add mapping from icons to coordinates (based on game environment)
        self.icon_names = ["dog", "cat", "bird", "cow", "sheep", "chicken", "piano", "trumpet", "drum", "flute"]
        
        # Set system prompt
        self.system_prompt = """
You are a professional AI assistant for a sound-based memory game.

Game Rules:
1. The game first plays an audiovisual sequence where each icon lights up and plays a corresponding sound.
2. Your task is to remember the order of the sequence.
3. Then, repeat the sequence by clicking the icons in the same order.
4. Icons include animals (dog, cat, bird, cow, sheep, chicken) and musical instruments (piano, trumpet, drum, flute).

Input Information:
1. Video – shows the sequence being played, with icons lighting up in order.
2. Audio – plays the sound associated with each icon in the sequence.
3. Screenshot – shows the current layout of the icons on the game interface.

Your Task:
1. Watch the video and listen to the audio to memorize the order and position of each icon in the sequence.
2. Analyze the game interface screenshot to identify the position of each icon.
3. Based on your memory of the sequence, provide the coordinates for which icon should be clicked next.

Coordinate System:
- Icons are arranged in a grid, starting from the top-left corner.
- Rows and columns are both 1-indexed.
- For example: the icon in the first row and first column has the coordinate (1, 1).
- The icon in the second row and third column has the coordinate (2, 3).
"""

    def encode_file_to_base64(self, file_path):
        """Encode a file to base64"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def extract_frames_to_base64_from_file(self, video_file_path):
        """
        Extract one frame per 0.5 second from a video file and convert them to Base64
        """
        base64_frames = []

        try:
            # Load the video directly from the file path
            clip = VideoFileClip(video_file_path)
            
            # Get the duration of the video in seconds
            duration = clip.duration
            
            # Use numpy.arange to generate timestamps at 0.5-second intervals
            timestamps = np.arange(0, duration, 0.5)
            
            # Extract frames at each timestamp
            for t in timestamps:
                # (H, W ,C)
                frame = clip.get_frame(t) 
                
                # Convert the frame to a PIL Image
                image = Image.fromarray(frame.astype('uint8'))
                
                # Create a BytesIO object to hold the image data
                buffered = io.BytesIO()
                image.save(buffered, format='JPEG')
                
                # Encode the image data to Base64
                base64_frame = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_frames.append(base64_frame)
                
            clip.close()
        
        except Exception as e:
            print(f"Error processing video {video_file_path}: {e}")
            
        return base64_frames

    def save_model_input_data(self, messages, description=""):
        """Save input data sent to model locally"""
        try:
            self.input_counter += 1
            input_dir = os.path.join(self.model_input_dir, f"input_{self.input_counter:03d}_{description}")
            os.makedirs(input_dir, exist_ok=True)
            
            # Save message content to JSON file
            messages_file = os.path.join(input_dir, "messages.json")
            with open(messages_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            
            # Extract and save various media files
            for i, message in enumerate(messages):
                if message.get("role") == "user" and isinstance(message.get("content"), list):
                    for j, content_item in enumerate(message["content"]):
                        # Save text content
                        if content_item.get("type") == "text":
                            text_file = os.path.join(input_dir, f"text_{i}_{j}.txt")
                            with open(text_file, 'w', encoding='utf-8') as f:
                                f.write(content_item["text"])
                        
                        # Save images
                        elif content_item.get("type") == "image_url":
                            image_data = content_item["image_url"]["url"]
                            if image_data.startswith("data:image/"):
                                # Extract base64 data
                                header, base64_data = image_data.split(",", 1)
                                image_bytes = base64.b64decode(base64_data)
                                
                                # Determine file extension
                                if "jpeg" in header:
                                    ext = "jpg"
                                elif "png" in header:
                                    ext = "png"
                                else:
                                    ext = "jpg"
                                
                                image_file = os.path.join(input_dir, f"image_{i}_{j}.{ext}")
                                with open(image_file, 'wb') as f:
                                    f.write(image_bytes)
                        
                        # Save audio - updated for input_audio format
                        elif content_item.get("type") == "input_audio":
                            audio_data = content_item["input_audio"]["data"]
                            audio_bytes = base64.b64decode(audio_data)
                            
                            # Use format field to determine extension
                            audio_format = content_item["input_audio"].get("format", "wav")
                            ext = audio_format
                            
                            audio_file = os.path.join(input_dir, f"audio_{i}_{j}.{ext}")
                            with open(audio_file, 'wb') as f:
                                f.write(audio_bytes)
            
            print(f"Model input data saved to: {input_dir}")
            return input_dir
            
        except Exception as e:
            print(f"Error saving model input data: {e}")
            return None

    def call_api(self, messages, description=""):
        """Call the API"""

        # Save input data locally
        self.save_model_input_data(messages, description)
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Build payload with multimodal support (video configuration removed)
        payload = {
            "model": MODEL_CHAT,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"voice": "Cherry", "format": "wav"},
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API request failed: {response.status_code}")
                print(f"Error message: {response.text}")
                return None
                
        except Exception as e:
            print(f"API call error: {e}")
            return None

    def prepare_multimodal_message(self, user_query, image_path=None, video_path=None, audio_path=None):
        """Prepare multimodal message"""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        content = [{"type": "text", "text": user_query}]
        
        # Add static image
        #if image_path and os.path.exists(image_path):
        #    image_b64 = self.encode_file_to_base64(image_path)
        #    content.append({
        #        "type": "image_url",
        #        "image_url": {
        #            "url": f"data:image/jpeg;base64,{image_b64}"
        #        }
        #    })
        
        # Add video frame sequence (replacing original video input)
        if video_path and os.path.exists(video_path):
            print(f"Extracting frames from video: {video_path}")
            video_frames = self.extract_frames_to_base64_from_file(video_path)
            
            if video_frames:
                # Add frame sequence description
                frame_description = f"The following {len(video_frames)} images show the sequence of the game being played (one frame every 0.5 seconds):"
                content.append({"type": "text", "text": frame_description})
                
                # Add each frame image
                for i, frame_b64 in enumerate(video_frames):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}"
                        }
                    })
                
                print(f"Added {len(video_frames)} video frames to message")
            else:
                print("Video frame extraction failed, skipping video input")
        
        messages.append({"role": "user", "content": content})
        
        # Add audio - using input_audio format
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as audio_file:
                audio = base64.b64encode(audio_file.read()).decode("utf-8")
            messages[1]["content"].append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio,
                    "format": "wav",
                },
            })
        
        return messages

    def capture_screen(self):
        """Capture the current screen"""
        screen = self.env.screen
        screen_array = pygame.surfarray.array3d(screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        return screen_array

    def get_icon_at_position(self, row, col):
        """Get icon name based on position"""
        try:
            # Calculate index in shapes list
            icon_index = row * self.env.cols + col
            
            # Check if index is valid
            if 0 <= icon_index < len(self.env.shapes):
                # Get icon name from environment's audio paths
                if hasattr(self.env, 'sound_manager') and hasattr(self.env.sound_manager, 'sound_paths'):
                    if icon_index < len(self.env.sound_manager.sound_paths):
                        sound_path = self.env.sound_manager.sound_paths[icon_index]
                        import os
                        filename = os.path.basename(sound_path)
                        icon_name = os.path.splitext(filename)[0].lower()
                        return icon_name
                
                # Fallback: use icon paths
                elif hasattr(self.env, 'icon_paths') and icon_index < len(self.env.icon_paths):
                    icon_path = self.env.icon_paths[icon_index]
                    import os
                    filename = os.path.basename(icon_path)
                    icon_name = os.path.splitext(filename)[0].lower()
                    return icon_name
                
                # Final fallback
                else:
                    if icon_index < len(self.icon_names):
                        return self.icon_names[icon_index]
                    else:
                        return "unknown"
            else:
                print(f"Icon index out of range: {icon_index}, shapes length: {len(self.env.shapes)}")
                return "unknown"
            
        except Exception as e:
            print(f"Error getting icon name: {e}")
            return "unknown"

    def extract_sequence_and_icons_from_response(self, response):
        """Extract sequence coordinates and icon information from model response"""
        try:
            sequence_coords = None
            sequence_icons = None
            
            # Extract sequence coordinates
            sequence_match = re.search(r"Sequence analysis:\s*\[([^\]]+)\]", response, re.IGNORECASE | re.DOTALL)
            if sequence_match:
                sequence_text = sequence_match.group(1)
                coord_matches = re.findall(r"\((\d+),(\d+)\)", sequence_text)
                if coord_matches:
                    sequence_coords = [(int(r), int(c)) for r, c in coord_matches]
            
            # Extract sequence icons
            icons_match = re.search(r"Sequence icons:\s*\[([^\]]+)\]", response, re.IGNORECASE | re.DOTALL)
            if icons_match:
                icons_text = icons_match.group(1)
                # Extract icon names within quotes
                icon_matches = re.findall(r"['\"]([^'\"]+)['\"]", icons_text)
                if icon_matches:
                    sequence_icons = [icon.strip() for icon in icon_matches]
                else:
                    # Try extracting icon names without quotes
                    icon_matches = re.findall(r"([a-zA-Z]+)", icons_text)
                    if icon_matches:
                        sequence_icons = [icon.strip() for icon in icon_matches]
            
            return sequence_coords, sequence_icons
            
        except Exception as e:
            print(f"Error extracting sequence and icon information: {e}")
            return None, None

    def calculate_coordinate_accuracy(self, predicted_coords, actual_coords):
        """Calculate accuracy of coordinate prediction"""
        if not predicted_coords or not actual_coords:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_coords), len(actual_coords))
        
        for i in range(min_length):
            if predicted_coords[i] == actual_coords[i]:
                correct_count += 1
            else:
                # Stop counting if the order is wrong
                break  
        
        return correct_count

    def calculate_icon_accuracy(self, predicted_icons, actual_icons):
        """Calculate accuracy of icon prediction"""
        if not predicted_icons or not actual_icons:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_icons), len(actual_icons))
        
        for i in range(min_length):
            if predicted_icons[i] == actual_icons[i]:
                correct_count += 1
            else:
                # Stop counting if the order is wrong
                break  
        
        return correct_count

    def prepare_ground_truth_data(self):
        """Prepare correct sequence data (coordinates and icons)"""
        try:
            # Get actual sequence coordinates
            actual_coords = []
            actual_icons = []
            
            print(f"Environment sequence info: {self.env.sequence}")
            print(f"Environment layout: {self.env.rows} rows x {self.env.cols} columns")
            
            for i, shape_index in enumerate(self.env.sequence):
                # sequence stores indices of shapes array
                if isinstance(shape_index, int) and 0 <= shape_index < len(self.env.shapes):
                    # Convert shape index to row-column coordinates
                    row = shape_index // self.env.cols
                    col = shape_index % self.env.cols
                    # Convert to 1-based coordinates
                    actual_coords.append((row + 1, col + 1))  
                    
                    # Get corresponding icon name
                    icon_name = self.get_icon_at_position(row, col)
                    actual_icons.append(icon_name)
                    
                    print(f"Sequence position {i}: shape_index={shape_index} -> coordinate=({row+1},{col+1}) -> icon={icon_name}")
                else:
                    print(f"Invalid sequence index: {shape_index}, type: {type(shape_index)}")
                    actual_coords.append((1, 1))
                    actual_icons.append("unknown")
            
            print(f"Final correct coordinate sequence: {actual_coords}")
            print(f"Final correct icon sequence: {actual_icons}")
            
            return actual_coords, actual_icons
            
        except Exception as e:
            print(f"Error preparing correct sequence data: {e}")
            print(f"Error details:")
            print(f"  Environment sequence: {getattr(self.env, 'sequence', 'None')}")
            print(f"  Environment shapes count: {len(getattr(self.env, 'shapes', []))}")
            print(f"  Environment rows x cols: {getattr(self.env, 'rows', 'None')} x {getattr(self.env, 'cols', 'None')}")
            
            # Return default data based on sequence length
            sequence_len = len(getattr(self.env, 'sequence', []))
            default_coords = [(1, 1)] * sequence_len
            default_icons = ["unknown"] * sequence_len
            return default_coords, default_icons

    def analyze_sequence_with_model(self, video_path, audio_path):
        """Analyze the sequence with the model"""
        try:
            # Save current screen capture to project directory
            screen_array = self.capture_screen()
            screen_image = Image.fromarray(screen_array)
            screen_image_path = os.path.join(self.temp_dir, "screen_capture.jpg")
            screen_image.save(screen_image_path)
            
            if self.difficulty == 1:
                layout_description = "6 icons, arranged in 2 rows and 3 columns"
            elif self.difficulty == 2:
                layout_description = "10 icons, arranged in 2 rows and 5 columns"
            elif self.difficulty == 3:
                layout_description = "15 icons, arranged in 3 rows and 5 columns"
                        
            user_query = f"""   
            Please analyze the sequence in this rhythm memory game:
                        
            1. Look at the frame sequence images, which show the icons lighting up in a specific order over time.
            2. Listen to the audio, as the sounds also provide information about the sequence.
            3. Memorize the exact order in which the icons appear in the sequence.
                        
            Game layout: {layout_description}

            Available icons include: dog, cat, bird, cow, sheep, chicken, piano, trumpet, drum, flute.

            Please carefully observe the lighting sequence in the frame images and listen to the order of sounds in the audio to accurately remember the sequence.

            The sequence length must match the total number of icons in the layout
            For 6 icons: sequence length = 6, for 10 icons: sequence length = 10, for 15 icons: sequence length = 15

            [IMPORTANT!!!] Your response must end in one of the following formats:
            For sequence analysis:
            Sequence analysis: [(row1,col1) (row2,col2) ...(rowN,colN)]
            Sequence icons: [icon1, icon2, ..., iconN]
            """
            
            messages = self.prepare_multimodal_message(
                user_query,
                image_path=screen_image_path if os.path.exists(screen_image_path) else None,
                video_path=video_path,
                audio_path=audio_path
            )
            
            print("Starting sequence analysis...")
            model_response = self.call_api(messages, "sequence_analysis")
            
            if model_response is None:
                print("Sequence analysis failed")
                self.stats["sequence_analysis_errors"] += 1
                return None, None, None
            
            print("-" * 60)
            print("Model sequence analysis result:", model_response)
            print("-" * 60)
            
            # Extract sequence coordinates and icon information from response
            predicted_coords, predicted_icons = self.extract_sequence_and_icons_from_response(model_response)
            
            if predicted_coords is None:
                print("Unable to parse sequence coordinates response")
                self.stats["unparseable_sequences"] += 1
                return None, None, model_response
            
            return predicted_coords, predicted_icons, model_response
            
        except Exception as e:
            print(f"Sequence analysis error: {e}")
            self.stats["sequence_analysis_errors"] += 1
            return None, None, None

    def get_click_suggestion(self, sequence_info, current_step, last_click_feedback_dir):
        """Get click suggestion from the model"""
        try:
            user_query = f"""
Now it's time to repeat the remembered sequence.

Current status:
- Sequence information: {sequence_info}
- You will be given the feedback from the previous click (highlighted image and sound).

Your task:
1. Analyze the image and audio feedback from the previous click.
2. Based on your memory of the original sequence, determine which icon comes next.
3. Provide the correct row and column coordinates for the next click.

Note:
- The icon grid starts from the top-left corner.
- Coordinates are 1-based (i.e., row 1, column 1 is written as (1,1)).

[IMPORTANT!!!] Your response must end with one of the following formats:
For click suggestion:
Suggested click: (row,col)
Example: Suggested click: (1,2) or Suggested click: (2,3)
"""

            click_frame_path = None
            click_audio_path = None
            
            if last_click_feedback_dir:
                click_frame_path = os.path.join(last_click_feedback_dir, "click_frame.png")
                click_audio_path = os.path.join(last_click_feedback_dir, "click_audio.wav")
            
            messages = self.prepare_multimodal_message(
                user_query,
                image_path=click_frame_path if click_frame_path and os.path.exists(click_frame_path) else None,
                audio_path=click_audio_path if click_audio_path and os.path.exists(click_audio_path) else None
            )
            
            print(f"Getting click suggestion for step {current_step + 1}...")
            model_response = self.call_api(messages, f"click_suggestion_step_{current_step + 1}")
            
            if model_response is None:
                print("Failed to get click suggestion")
                return None
            
            print("Model click position selection:")
            print("-" * 40)
            print(model_response)

            click_coord = self.extract_click_from_response(model_response)
            return click_coord
            
        except Exception as e:
            print(f"Error getting click suggestion: {e}")
            self.stats["click_prediction_errors"] += 1
            return None
    
    
    def extract_sequence_from_response(self, response):
        """Extract sequence information from model response (for backward compatibility)"""
        try:
            predicted_coords, _ = self.extract_sequence_and_icons_from_response(response)
            return predicted_coords
        except Exception as e:
            print(f"Error extracting sequence information: {e}")
            return None



    def extract_click_from_response(self, response):
        """Extract click coordinates from model response"""
        try:
            # Standard format matching: "Suggested click: (1, 2)"
            click_match = re.search(r"Suggested click:\s*\(?\s*(\d+)\s*,?\s+(\d+)\s*\)?", response, re.IGNORECASE)
            if click_match:
                row = int(click_match.group(1))
                col = int(click_match.group(2))
                # Convert to 0-based index
                return (row - 1, col - 1)  
            
            # Try other formats
            coord_match = re.search(r"\((\d+),(\d+)\)", response)
            if coord_match:
                row = int(coord_match.group(1))
                col = int(coord_match.group(2))
                # Convert to 0-based index
                return (row - 1, col - 1)  
            
            print("Failed to extract click coordinates from response")
            self.stats["click_prediction_errors"] += 1
            return None
            
        except Exception as e:
            print(f"Error extracting click coordinate: {e}")
            self.stats["click_prediction_errors"] += 1
            return None

    def save_stats(self, round_num):
        """Save statistics to file"""
        save_dir = pathlib.Path("./rhythm_memory_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        script_name = os.path.basename(__file__).replace('.py', '')
        
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("===== Rhythm Memory Game Statistics =====\n")
            f.write(f"Difficulty Level: {self.difficulty}\n")
            f.write(f"Round: {self.stats['rounds']}\n")
            f.write(f"Successful Rounds: {self.stats['successful_rounds']}\n")
            f.write(f"Total Score: {self.stats['total_score']}\n")
            f.write(f"Sequence Analysis Errors: {self.stats['sequence_analysis_errors']}\n")
            f.write(f"Click Prediction Errors: {self.stats['click_prediction_errors']}\n")
            f.write(f"Save Time: {timestamp}\n")
        
        print(f"Statistics saved to: {filepath}")


    def run_single_round(self, round_num):
        """Run a single round of the game"""
        print(f"\n===== Starting Round {round_num} =====")
        
        observation = self.env.reset()
        self.env.render()
        round_successful = False
        
        try:
            # Phase 1: Play Sequence
            print("--- Play Sequence Phase ---")
            # Special action to start playback
            action = (self.env.rows, self.env.cols)  
            obs, reward, done, info = self.env.step(action)
            time.sleep(2)
            # Get the generated video and audio files
            video_path = os.path.join(self.env.record_dir, "sequence_video.mp4")    
            audio_path = os.path.join(self.env.record_dir, "sequence_audio.wav")
            max_wait = 10
            wait_time = 0
            while not os.path.exists(video_path) and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            # Analyze sequence
            predicted_coords, predicted_icons, _ = self.analyze_sequence_with_model(video_path, audio_path)    
            if predicted_coords is None:
                print("Sequence analysis failed")
                return False
            
            # Phase 2: Repeat Sequence
            print("\n--- Repeat Sequence Phase ---")
            current_step = 0
            while not done and current_step < len(self.env.sequence):
                if current_step == 0:
                    # First step directly uses the sequence analyzed by the model, no need for the model to predict from the image again
                    coord = predicted_coords[0]
                    click_coord = (coord[0] - 1, coord[1] - 1)
                    print(f"(Step 1) Directly using the analyzed click position: {click_coord}")
                else:
                    click_coord = self.get_click_suggestion(predicted_coords, current_step, last_click_feedback_dir=self.env.record_dir)
                    if click_coord is None:
                        print("Could not get click suggestion")
                        break

                print(f"Suggested click index: {click_coord}")
                obs, reward, done, info = self.env.step(click_coord)
                self.env.render()
                print(f"Step {current_step + 1}, Reward: {reward}, Progress: {obs['progress']}/{len(self.env.sequence)}")
                current_step += 1
                # Brief pause
                time.sleep(0.5)         
            
            # Check for success
            if obs['progress'] == len(self.env.sequence):
                print("🎉 Round successfully completed!")
                round_successful = True
                self.stats["successful_rounds"] += 1
                self.stats["total_score"] += obs['progress']
            else:
                print("😞 Round failed")
                self.stats["total_score"] += obs['progress']
            
        except Exception as e:
            print(f"Round execution error: {e}")
        
        self.stats["rounds"] += 1
        return round_successful


    def calculate_correct_predictions(self, predicted_sequence, actual_sequence):
        """Calculate the number of correctly predicted icons and coordinates"""
        if not predicted_sequence or not actual_sequence:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_sequence), len(actual_sequence))
        
        for i in range(min_length):
            if predicted_sequence[i] == actual_sequence[i]:
                correct_count += 1
            else:
                # Stop counting once an error is found, as the order must be correct
                break  
        
        return correct_count

    def save_episode_results(self):
        """Save all episode results to file"""
        save_dir = pathlib.Path("./rhythm_memory_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # Save detailed JSON results
        json_filename = f"{script_name}_detailed_results_{timestamp}.json"
        json_filepath = save_dir / json_filename
        
        detailed_results = {
            "experiment_info": {
                "difficulty": self.difficulty,
                "total_episodes": self.stats["total_episodes"],
                "timestamp": timestamp,
                "script_name": script_name
            },
            "overall_stats": self.stats,
            "episodes": self.stats["episodes_data"]
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # Save a brief text summary
        txt_filename = f"{script_name}_summary_{timestamp}.txt"
        txt_filepath = save_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("===== Rhythm Memory Game Statistics Report =====\n")
            f.write(f"Difficulty Level: {self.difficulty}\n")
            f.write(f"Total Episodes: {self.stats['total_episodes']}\n")
            f.write(f"Successful Episodes: {self.stats['successful_episodes']}\n")
            f.write(f"Success Rate: {(self.stats['successful_episodes']/max(self.stats['total_episodes'], 1))*100:.1f}%\n")
            f.write(f"Total Score: {self.stats['total_score']}\n")
            f.write(f"Average Score: {self.stats['total_score']/max(self.stats['total_episodes'], 1):.2f}\n")
            f.write(f"Sequence Analysis Errors: {self.stats['sequence_analysis_errors']}\n")
            f.write(f"Click Prediction Errors: {self.stats['click_prediction_errors']}\n")
            f.write(f"Unparseable Sequences: {self.stats['unparseable_sequences']}\n")
            f.write(f"Save Time: {timestamp}\n\n")
            
            f.write("===== Detailed Results for Each Episode =====\n")
            for ep_data in self.stats["episodes_data"]:
                f.write(f"Episode {ep_data['episode_num']}: ")
                f.write(f"Success={ep_data['success']}, ")
                f.write(f"Score={ep_data['score']}, ")
                f.write(f"Correct Coordinates={ep_data['correct_coordinates']}/{ep_data['sequence_length']}, ")
                f.write(f"Correct Icons={ep_data['correct_icons']}/{ep_data['sequence_length']}, ")
                f.write(f"Sequence Parseable={ep_data['sequence_parseable']}\n")
        
        print(f"Detailed results saved to: {json_filepath}")
        print(f"Summary statistics saved to: {txt_filepath}")
        return json_filepath, txt_filepath

    def run_single_episode(self, episode_num):
        """Run a single episode of the game"""
        print(f"\n===== Episode {episode_num}/{self.max_episodes} =====")
        
        # Initialize episode data
        episode_data = {
            "episode_num": episode_num,
            "success": False,
            "score": 0,
            "sequence_length": 0,
            "correct_coordinates": 0,
            "correct_icons": 0,
            "sequence_parseable": False,
            "actual_coordinates": [],
            "actual_icons": [],
            "predicted_coordinates": [],
            "predicted_icons": [],
            "model_raw_response": "",
            "error_type": None
        }
        
        observation = self.env.reset()
        self.env.render()
        
        try:
            # Phase 1: Play Sequence
            print("--- Play Sequence Phase ---")
            # Special action to start playback
            action = (self.env.rows, self.env.cols)  
            obs, reward, done, info = self.env.step(action)
            time.sleep(2)
            
            # Prepare ground truth data
            actual_coordinates, actual_icons = self.prepare_ground_truth_data()
            episode_data["actual_coordinates"] = actual_coordinates
            episode_data["actual_icons"] = actual_icons
            episode_data["sequence_length"] = len(actual_coordinates)
            
            print(f"Correct coordinate sequence: {actual_coordinates}")
            print(f"Correct icon sequence: {actual_icons}")
            
            video_path = os.path.join(self.env.record_dir, "sequence_video.mp4")
            audio_path = os.path.join(self.env.record_dir, "sequence_audio.wav")
            max_wait = 10
            wait_time = 0
            while not os.path.exists(video_path) and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            # Analyze sequence
            predicted_coords, predicted_icons, raw_response = self.analyze_sequence_with_model(video_path, audio_path)
            
            if predicted_coords is None:
                print("Sequence analysis failed, continuing game with a random strategy")
                episode_data["sequence_parseable"] = False
                episode_data["error_type"] = "sequence_analysis_failed"
                # Generate a random sequence as a fallback
                predicted_coords = [(1, 1)] * len(self.env.sequence)
                predicted_icons = ["unknown"] * len(self.env.sequence)
            else:
                episode_data["sequence_parseable"] = True
                episode_data["model_raw_response"] = raw_response or ""
            
            episode_data["predicted_coordinates"] = predicted_coords
            episode_data["predicted_icons"] = predicted_icons or []
            
            print(f"Predicted coordinate sequence: {predicted_coords}")
            print(f"Predicted icon sequence: {predicted_icons}")
            
            # Calculate accuracy
            episode_data["correct_coordinates"] = self.calculate_coordinate_accuracy(
                predicted_coords, actual_coordinates
            )
            episode_data["correct_icons"] = self.calculate_icon_accuracy(
                predicted_icons, actual_icons
            )
            
            print(f"Coordinate prediction accuracy: {episode_data['correct_coordinates']}/{episode_data['sequence_length']}")
            print(f"Icon prediction accuracy: {episode_data['correct_icons']}/{episode_data['sequence_length']}")
            
            # Phase 2: Repeat Sequence
            print("\n--- Repeat Sequence Phase ---")
            current_step = 0
            while not done and current_step < len(self.env.sequence):
                if current_step == 0:
                    # First step directly uses the sequence from model analysis
                    if predicted_coords and len(predicted_coords) > 0:
                        coord = predicted_coords[0]
                        click_coord = (coord[0] - 1, coord[1] - 1)
                    else:
                        # If sequence parsing fails, use a random click
                        click_coord = (0, 0)
                    print(f"(Step 1) Using click position: {click_coord}")
                else:
                    if episode_data["sequence_parseable"]:
                        click_coord = self.get_click_suggestion(predicted_coords, current_step, 
                                                              last_click_feedback_dir=self.env.record_dir)
                    else:
                        # If sequence is not parseable, use a random strategy
                        click_coord = (current_step % self.env.rows, current_step % self.env.cols)
                    
                    if click_coord is None:
                        print("Could not get click suggestion, using a random click")
                        click_coord = (current_step % self.env.rows, current_step % self.env.cols)

                print(f"Clicking coordinate: {click_coord}")
                obs, reward, done, info = self.env.step(click_coord)
                self.env.render()
                print(f"Step {current_step + 1}, Reward: {reward}, Progress: {obs['progress']}/{len(self.env.sequence)}")
                current_step += 1
                time.sleep(0.5)
            
            # Record final result
            episode_data["score"] = obs['progress']
            if obs['progress'] == len(self.env.sequence):
                print("🎉 Episode successfully completed!")
                episode_data["success"] = True
                self.stats["successful_episodes"] += 1
            else:
                print("😞 Episode failed")
            
            self.stats["total_score"] += obs['progress']
            
        except Exception as e:
            print(f"Episode execution error: {e}")
            episode_data["error_type"] = f"execution_error: {str(e)}"
        
        # Add episode data to statistics
        self.stats["episodes_data"].append(episode_data)
        self.stats["total_episodes"] += 1
        
        # Print current episode statistics
        print(f"Episode {episode_num} results:")
        print(f"  Success: {episode_data['success']}")
        print(f"  Score: {episode_data['score']}/{episode_data['sequence_length']}")
        print(f"  Correct Coordinates: {episode_data['correct_coordinates']}/{episode_data['sequence_length']}")
        print(f"  Correct Icons: {episode_data['correct_icons']}/{episode_data['sequence_length']}")
        print(f"  Sequence Parseable: {episode_data['sequence_parseable']}")
        
        return episode_data["success"]

    def run(self):
        """Run the AI test"""
        print(f"Starting Rhythm Memory Game AI Test - Difficulty: {self.difficulty}")
        print(f"Will run {self.max_episodes} episodes")
        print(f"Data save path: {self.temp_dir}")
        
        # Check API connection
        try:
            test_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            test_response = self.call_api(test_messages, "connection_test")
            if test_response:
                print("✅ API connection is normal")
            else:
                print("❌ API connection is abnormal")
                return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return

        start_time = time.time()
        
        try:
            for episode_num in range(1, self.max_episodes + 1):
                success = self.run_single_episode(episode_num)
                
                # Print progress
                current_success_rate = (self.stats["successful_episodes"] / self.stats["total_episodes"]) * 100
                print(f"Progress: {episode_num}/{self.max_episodes}, Current Success Rate: {current_success_rate:.1f}%")
                
                # Brief pause between episodes
                if episode_num < self.max_episodes:
                    time.sleep(1)
                
                # Save intermediate results every 10 episodes
                if episode_num % 10 == 0:
                    print(f"Completed {episode_num} episodes, saving intermediate results...")
                    self.save_episode_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print final statistics
            print("\n" + "="*60)
            print("===== Final Statistics ======")
            print(f"Total episodes: {self.stats['total_episodes']}")
            print(f"Successful episodes: {self.stats['successful_episodes']}")
            print(f"Success Rate: {(self.stats['successful_episodes']/max(self.stats['total_episodes'], 1))*100:.1f}%")
            print(f"Total score: {self.stats['total_score']}")
            print(f"Average score: {self.stats['total_score']/max(self.stats['total_episodes'], 1):.2f}")
            print(f"Sequence analysis errors: {self.stats['sequence_analysis_errors']}")
            print(f"Click prediction errors: {self.stats['click_prediction_errors']}")
            print(f"Unparseable sequences: {self.stats['unparseable_sequences']}")
            print(f"Total time elapsed: {total_time:.1f} seconds")
            print("="*60)
            
            # Save final results
            json_file, txt_file = self.save_episode_results()
            
        except KeyboardInterrupt:
            print("\nUser interrupted the test")
            print("Saving current results...")
            self.save_episode_results()
        finally:
            # Clean up resources
            self.env.close()
            pygame.quit()
            
            print(f"\nModel input data has been saved to: {self.model_input_dir}")
            print(f"All data saved in directory: {self.temp_dir}")
            print("\nRhythm Memory Game AI Test Finished!")

def select_difficulty():
    """Select game difficulty"""
    print("Please select the game difficulty:")
    print("1. Easy - 6 icons, 2 rows x 3 columns")
    print("2. Normal - 10 icons, 2 rows x 5 columns")
    print("3. Hard - 15 icons, 3 rows x 5 columns")
    
    while True:
        choice = input("Enter your choice (1-3): ")
        if choice in ["1", "2", "3"]:
            return int(choice)
        else:
            print("Invalid choice, please enter again.")


if __name__ == "__main__":
    # Select difficulty
    # difficulty = select_difficulty()
    difficulty = 3
    max_episodes = 10
    runner = ModelRhythmMemoryRunner(difficulty=difficulty, max_episodes=max_episodes)
    runner.run()
    runner = ModelRhythmMemoryRunner(difficulty=difficulty, max_episodes=max_episodes)
    runner.run()