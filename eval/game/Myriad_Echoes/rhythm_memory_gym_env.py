import gym
from gym import spaces
import pygame
import random
import numpy as np
import sys
from moviepy.editor import ImageSequenceClip
from pydub import AudioSegment
import os
import glob
os.environ["SDL_AUDIODRIVER"] = "dummy"

base_dir = os.path.dirname(os.path.abspath(__file__))

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SHAPE_SIZE = 100
SHAPE_GAP = 40
GLOW_RADIUS = 18

class SoundManager:
    def __init__(self, sound_files):
        pygame.mixer.init()
        self.sounds = [pygame.mixer.Sound(f) for f in sound_files]
        self.sound_paths = sound_files  # Save paths

    def play_sound(self, index, volume=0.5, save_path=None):
        if 0 <= index < len(self.sounds):
            snd = self.sounds[index]
            snd.set_volume(volume)
            snd.play()

            if save_path:
                from shutil import copyfile
                copyfile(self.sound_paths[index], save_path)

            return snd.get_length(), self.sound_paths[index]  # ✅ Return path
        return 1.0, None


class GameElement:
    def __init__(self, image_path, sound_index, center):
        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.smoothscale(self.image, (SHAPE_SIZE, SHAPE_SIZE))
        self.center = center
        self.rect = self.image.get_rect(center=center)
        self.sound_index = sound_index
        self.glow = 0.0

    def draw(self, surface):
        if self.glow > 0:
            for i in range(GLOW_RADIUS, 0, -4):
                alpha = int(60 * self.glow * (i / GLOW_RADIUS))
                glow_surf = pygame.Surface((SHAPE_SIZE + i*2, SHAPE_SIZE + i*2), pygame.SRCALPHA)
                pygame.draw.ellipse(
                    glow_surf,
                    (255, 230, 150, alpha),
                    (0, 0, SHAPE_SIZE + i*2, SHAPE_SIZE + i*2)
                )
                glow_rect = glow_surf.get_rect(center=self.center)
                surface.blit(glow_surf, glow_rect)
        surface.blit(self.image, self.rect)


class RhythmMemoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, difficulty):
        super().__init__()

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Rhythm Memory Gym")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 28, bold=True)
        self.small_font = pygame.font.SysFont('arial', 20)

        self.difficulty = difficulty
        self.lives = 1
        self.cumulative_reward = 0
        
        self.record_dir = f"{base_dir}/assets/temp"
        os.makedirs(self.record_dir, exist_ok=True)
        for f in glob.glob(os.path.join(self.record_dir, "*")):
            os.remove(f)
        self.sequence_frames = []         # Save frame paths
        self.sequence_audio_segments = [] # Save audio segments (pydub)

        self.icon_paths = [
            f"{base_dir}/assets/icons/Dog.png",
            f"{base_dir}/assets/icons/Cat.png",
            f"{base_dir}/assets/icons/bird.png",
            f"{base_dir}/assets/icons/cow.png",
            f"{base_dir}/assets/icons/sheep.png",
            f"{base_dir}/assets/icons/chicken.png",
            f"{base_dir}/assets/icons/piano.png",
            f"{base_dir}/assets/icons/trumpet.png",
            f"{base_dir}/assets/icons/drum.png",
            f"{base_dir}/assets/icons/flute.png"
        ]

        self.sound_paths = [
            f"{base_dir}/assets/wav_files/dog.wav",
            f"{base_dir}/assets/wav_files/cat.wav",
            f"{base_dir}/assets/wav_files/bird.wav",
            f"{base_dir}/assets/wav_files/cow.wav",
            f"{base_dir}/assets/wav_files/sheep.wav",
            f"{base_dir}/assets/wav_files/chicken.wav",
            f"{base_dir}/assets/wav_files/piano.wav",
            f"{base_dir}/assets/wav_files/trumpet.wav",
            f"{base_dir}/assets/wav_files/drum.wav",
            f"{base_dir}/assets/wav_files/flute.wav"
        ]

        self.shapes = []
        self.sequence = []
        self.sequence_progress = 0

        self.state = 'show_sequence'
        self.play_index = 0

        if difficulty == 1:      # Easy: 6 icons
            self.num_shapes = 6
            self.rows = 2
            self.cols = 3
        elif difficulty == 2:    # Normal: 10 unique icons
            self.num_shapes = 10
            self.rows = 2
            self.cols = 5
        elif difficulty == 3:    # Hard: 15 icons, may repeat
            self.num_shapes = 15
            self.rows = 3
            self.cols = 5

        self._setup_game()      #


    def _setup_game(self):      # Initialize sequence for a new round
        if self.difficulty < 3:
            selected_indices = random.sample(range(len(self.icon_paths)), self.num_shapes)
        else:
            selected_indices = [random.randint(0, len(self.icon_paths) - 1) for _ in range(self.num_shapes)]
        icons = [self.icon_paths[i] for i in selected_indices]
        sounds = [self.sound_paths[i] for i in selected_indices]
        self.sound_manager = SoundManager(sounds)

        self.shapes = []
        y_start = SCREEN_HEIGHT // 2 - SHAPE_SIZE - SHAPE_GAP // 2
        for i in range(self.num_shapes):
            row = i // self.cols
            col = i % self.cols
            total_width = self.cols * SHAPE_SIZE + (self.cols - 1) * SHAPE_GAP
            start_x = (SCREEN_WIDTH - total_width) // 2 + SHAPE_SIZE // 2
            x = start_x + col * (SHAPE_SIZE + SHAPE_GAP)
            y = y_start + row * (SHAPE_SIZE + SHAPE_GAP)
            self.shapes.append(GameElement(icons[i], i, (x, y)))

        if self.difficulty == 1:
            seq_len = 6
        elif self.difficulty == 2:
            seq_len = 10
        else:
            seq_len = 15
            
        self.sequence = random.sample(range(self.num_shapes), seq_len)
        self.sequence_progress = 0
        self.state = 'show_sequence'


    def step(self, action):     # action: (row, col), one game step, returns obs, reward, done, info
        reward = 0
        done = False
        info = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        row, col = action
        if self.state == 'show_sequence':       # Show sequence
            if (row, col) == (self.rows, self.cols):
                self.sequence_frames.clear()
                self.sequence_audio_segments.clear()
                print("Sequence :", self.sequence)

                for i, idx in enumerate(self.sequence):
                    frame_path = os.path.join(self.record_dir, f"sequence_frame_{i:02d}.png")   # --- Save frame ---
                    self._highlight_shape(idx)
                    pygame.image.save(self.screen, frame_path)
                    self.sequence_frames.append(frame_path)
                    pygame.time.delay(200)

                    _, sound_path = self.sound_manager.play_sound(idx)  # --- Play and save audio segment ---
                    audio = AudioSegment.from_file(sound_path)
                    self.sequence_audio_segments.append(audio)

                    pygame.time.delay(int(audio.duration_seconds * 1000))

                    for shape in self.shapes:
                        shape.glow = 0.0
                    self._render_shapes()
                    pygame.display.flip()
                    pygame.time.delay(120)

                full_audio = sum(self.sequence_audio_segments)      # --- Synthesize audio file ---
                full_audio.export(os.path.join(self.record_dir, "sequence_audio.wav"), format="wav")
                clip = ImageSequenceClip(self.sequence_frames, fps=0.8)     # --- Synthesize video file ---
                clip.write_videofile(os.path.join(self.record_dir, "sequence_video.mp4"))

                self.state = 'player_input'
                
                return self._get_obs(), 0, False, info


        if self.state == 'player_input':        # Player input
            if 0 <= row < self.rows and 0 <= col < self.cols:
                idx_clicked = row * self.cols + col
                print('current position:', self.sequence[self.sequence_progress], "model click at:", idx_clicked)
                if idx_clicked == self.sequence[self.sequence_progress]:
                    
                    self._highlight_shape(idx_clicked)    # --- Successful click, save frame ---
                    pygame.image.save(self.screen, os.path.join(self.record_dir, "click_frame.png"))

                    self.sound_manager.play_sound(idx_clicked, save_path=os.path.join(self.record_dir, "click_audio.wav"))  # --- Play and save sound ---

                    self.sequence_progress += 1
                    reward += 1
                    if self.sequence_progress == len(self.sequence):
                        done = True
                else:
                    done = True
            else:
                done = True

        self.cumulative_reward += reward
        return self._get_obs(), self.cumulative_reward, done, info

    def _highlight_shape(self, idx):
        for i, shape in enumerate(self.shapes):
            shape.glow = 1.0 if i == idx else 0.0
        self._render_shapes()
        pygame.display.flip()

    def _play_sound(self, idx):
        self.sound_manager.play_sound(idx)

    def _render_shapes(self):
        self.screen.fill((255, 255, 255))
        if self.state == 'show_sequence':
            title = self.small_font.render("Type 'play' to hear the sequence", True, (50, 50, 50))
            self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))
        for shape in self.shapes:
            shape.draw(self.screen)

    def _get_obs(self):
        return {
            'progress': self.sequence_progress,
        }

    def render(self, mode='human'):
        self._render_shapes()
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def reset(self):
        """Reset the game environment"""
        self.lives = 1
        self.cumulative_reward = 0
        self.sequence_progress = 0
        self.state = 'show_sequence'
        self.play_index = 0
        
        # Clean up previous recording files
        for f in glob.glob(os.path.join(self.record_dir, "*")):
            os.remove(f)
        self.sequence_frames = []
        self.sequence_audio_segments = []
        
        # Reset the game
        self._setup_game()
        
        return self._get_obs()

def show_intro_screen(screen, font, small_font):
    screen.fill((255, 255, 255))
    title = font.render("Rhythm Memory Master", True, (0, 0, 0))
    info = small_font.render("Enter 1/2/3 to select difficulty, then type 'play' to start", True, (50, 50, 50))
    screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, SCREEN_HEIGHT//4))
    screen.blit(info, (SCREEN_WIDTH//2 - info.get_width()//2, SCREEN_HEIGHT//4 + 60))

    # Draw all 10 icons centered below the text
    icon_size = 60
    icon_gap = 20
    icons = [
        pygame.image.load(p).convert_alpha() for p in [
            f"{base_dir}/assets/icons/Dog.png",
            f"{base_dir}/assets/icons/Cat.png",
            f"{base_dir}/assets/icons/bird.png",
            f"{base_dir}/assets/icons/cow.png",
            f"{base_dir}/assets/icons/sheep.png",
            f"{base_dir}/assets/icons/chicken.png",
            f"{base_dir}/assets/icons/piano.png",
            f"{base_dir}/assets/icons/trumpet.png",
            f"{base_dir}/assets/icons/drum.png",
            f"{base_dir}/assets/icons/flute.png"
        ]
    ]
    icons = [pygame.transform.smoothscale(icon, (icon_size, icon_size)) for icon in icons]
    icons_per_row = 5
    start_y = SCREEN_HEIGHT // 2 - icon_size - icon_gap // 2 + 100
    for i, icon in enumerate(icons):
        row = i // icons_per_row
        col = i % icons_per_row
        total_width = icons_per_row * icon_size + (icons_per_row - 1) * icon_gap
        start_x = (SCREEN_WIDTH - total_width) // 2
        x = start_x + col * (icon_size + icon_gap)
        y = start_y + row * (icon_size + icon_gap)
        screen.blit(icon, (x, y))

    pygame.display.flip()



if __name__ == '__main__':
    pass