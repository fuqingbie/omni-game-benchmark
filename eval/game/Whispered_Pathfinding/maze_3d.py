import pygame
import numpy as np
import os
import json
import math
import random
from pygame.locals import *

class MazeGame:
    # Move game configuration constants to the class level for easy access in the Gym environment
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    MAZE_SIZE = 8  # Base maze size, may increase on hard difficulty
    WALL_HEIGHT = 2.0
    FOV = math.pi / 3
    MAX_DEPTH = 16
    MOVE_SPEED = 0.08
    TURN_SPEED = math.pi / 30
    RAY_COUNT = 160
    
    # Difficulty settings
    DIFFICULTY_SETTINGS = {
        'easy': {
            'wall_remove_prob': 0.3,  # Probability of removing walls
            'additional_obstacles': 0,  # Number of additional obstacles
            'narrow_path_factor': 0.0,  # Path narrowing factor (0 means no narrowing)
            'dead_ends': 0,            # Number of dead ends
            'maze_size_increase': 0    # Maze size increase amount
        },
        'medium': {
            'wall_remove_prob': 0.05,
            'additional_obstacles': 12,
            'narrow_path_factor': 0.0,
            'dead_ends': 2,
            'maze_size_increase': 0
        },
        'hard': {
            'wall_remove_prob': 0.05,
            'additional_obstacles': 18,
            'narrow_path_factor': 0.25,  # Path width reduced by 25%
            'dead_ends': 5,
            'maze_size_increase': 2      # Maze size increases by 2, i.e., 10x10
        }
    }
    
    def __init__(self, difficulty='easy'):
        # Initialize Pygame
        pygame.init()
        pygame.mixer.init()
        
        # Set difficulty
        self.difficulty = difficulty if difficulty in self.DIFFICULTY_SETTINGS else 'easy'
        self.difficulty_config = self.DIFFICULTY_SETTINGS[self.difficulty]
        
        # Adjust maze size based on difficulty
        self.current_maze_size = self.MAZE_SIZE + self.difficulty_config['maze_size_increase']
        
        # Create window
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(f"3D Maze Adventure - {self.difficulty.capitalize()}")
        self.clock = pygame.time.Clock()
        
        # Generate maze
        self.maze = self.generate_simplified_maze(self.current_maze_size)
        
        # Player attributes
        self.player_pos = np.array([1.5, 1.5])
        self.player_angle = 0
        self.player_fov = self.FOV
        
        # Set exit position
        self.goal_pos = np.array([self.current_maze_size - 1.5, self.current_maze_size - 1.5])
        
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load sound effects
        try:
            sound_path = os.path.join(script_dir, '..', '..', 'assets-necessary', 'kenney', 'Audio', 'Voiceover Pack', 'Audio (Female)', 'go.ogg')
            if os.path.exists(sound_path):
                self.sound_go = pygame.mixer.Sound(sound_path)
            else:
                self.sound_go = None
        except:
            self.sound_go = None
            
        # Load goal icon texture
        try:
            texture_path = os.path.join(script_dir, '..', '..', 'assets-necessary', 'kenney', '3D assets', 'Holiday Kit', 'Previews', 'present-a-cube.png')
            if os.path.exists(texture_path):
                self.goal_texture = pygame.image.load(texture_path)
            else:
                self.goal_texture = None
        except:
            self.goal_texture = None
        
        # Ensure the goal texture is scaled correctly
        if self.goal_texture:
            goal_size = 128
            self.goal_texture = pygame.transform.scale(self.goal_texture, (goal_size, goal_size))
            if hasattr(self.goal_texture, 'convert_alpha'):
                self.goal_texture = self.goal_texture.convert_alpha()
        
        # Game state
        self.running = True
        self.won = False
        self.use_lighting = True
        self.show_minimap = True
        self.show_path_hints = True
        
        # Frame counter
        self.frame_counter = 0
        self.fps_values = []
        self.last_fps_time = pygame.time.get_ticks()
        self.current_fps = 0
        
        # Minimap cache
        self.minimap_surface = None
        self.minimap_update_counter = 0
    
    def load_sound(self, name):
        """Find and load sound by name"""
        matching_paths = [p for p in self.asset_paths if name in p]
        if matching_paths:
            try:
                return pygame.mixer.Sound(matching_paths[0])
            except:
                pass
        return None
    
    def generate_simplified_maze(self, size):
        """Generate a simplified maze, adjusted for difficulty"""
        # Create a maze filled with walls (1 is wall, 0 is path)
        maze = np.ones((size, size), dtype=np.int8)
        
        # Start generation from (1,1)
        def carve_path(x, y):
            maze[y][x] = 0  # Set as path
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < size-1 and 0 < ny < size-1 and maze[ny][nx] == 1:
                    maze[y + dy//2][x + dx//2] = 0
                    carve_path(nx, ny)
        
        carve_path(1, 1)
        
        # Ensure the entrance and exit are paths
        maze[1][1] = 0
        maze[size-2][size-2] = 0
        
        # Randomly remove walls based on difficulty
        wall_remove_prob = self.difficulty_config['wall_remove_prob']
        for y in range(1, size-1):
            for x in range(1, size-1):
                if maze[y][x] == 1 and random.random() < wall_remove_prob:
                    path_count = sum(maze[y+dy][x+dx] == 0 
                                     for dy in [-1, 0, 1] 
                                     for dx in [-1, 0, 1]
                                     if 0 <= x+dx < size and 0 <= y+dy < size)
                    if path_count >= 1:
                        maze[y][x] = 0
        
        # Create a direct path from start to exit
        self.create_path_to_exit(maze, 1, 1, size-2, size-2)
        
        # Add additional obstacles based on difficulty
        self.add_additional_obstacles(maze, size)
        
        return maze
    
    def add_additional_obstacles(self, maze, size):
        """Add additional obstacles based on difficulty"""
        num_obstacles = self.difficulty_config['additional_obstacles']
        if num_obstacles <= 0:
            return
        
        # If it's hard mode, use advanced maze obstacle generation
        if self.difficulty == 'hard':
            self.create_maze_obstacles(maze, size, num_obstacles)
            return
            
        # Normal difficulty obstacle addition (original logic)
        empty_cells = []
        for y in range(1, size-1):
            for x in range(1, size-1):
                # Skip areas near the start and goal
                if (abs(x - 1) + abs(y - 1) <= 2) or (abs(x - (size-2)) + abs(y - (size-2)) <= 2):
                    continue
                    
                if maze[y][x] == 0:  # If it's a path
                    empty_cells.append((x, y))
        
        # Randomly select positions to add obstacles
        random.shuffle(empty_cells)
        for i in range(min(num_obstacles, len(empty_cells))):
            x, y = empty_cells[i]
            # Check if a path still exists after placing the obstacle
            maze[y][x] = 1  # Temporarily set as wall
            
            # Ensure a path from start to goal still exists after placing the obstacle (using simple connectivity check)
            if not self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                maze[y][x] = 0  # If the path is blocked, remove the obstacle

    def create_maze_obstacles(self, maze, size, num_obstacles):
        """Create maze-style obstacle layout, ensuring a unique path exists"""
        # Step 1: Find a path from the start to the goal (this will be our guaranteed path)
        path = self.find_shortest_path(maze, (1, 1), (size-2, size-2))
        if not path:
            return  # If no path is found, do not add obstacles
        
        # Step 2: Mark all points on this path and their immediate surroundings; obstacles cannot be placed here
        protected_cells = set()
        for x, y in path:
            protected_cells.add((x, y))
            # Add a one-cell protection zone around it
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        protected_cells.add((nx, ny))
        
        # Step 3: Identify suitable points for obstacles, i.e., empty points not in the protected area
        candidate_cells = []
        for y in range(1, size-1):
            for x in range(1, size-1):
                # Skip areas near the start and goal
                if (abs(x - 1) + abs(y - 1) <= 2) or (abs(x - (size-2)) + abs(y - (size-2)) <= 2):
                    continue
                
                if maze[y][x] == 0 and (x, y) not in protected_cells:
                    # Check if there is enough space around to place continuous obstacles
                    neighbor_spaces = []
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        dx, dy = direction
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < size and 0 <= ny < size and 
                            maze[ny][nx] == 0 and (nx, ny) not in protected_cells):
                            neighbor_spaces.append((nx, ny))
                    
                    # Only select positions with at least one available neighbor as starting points
                    if neighbor_spaces:
                        candidate_cells.append((x, y, neighbor_spaces))
        
        # Step 4: Randomly select starting points and generate continuous obstacle patterns
        obstacles_added = 0
        random.shuffle(candidate_cells)
        
        while candidate_cells and obstacles_added < num_obstacles:
            x, y, neighbors = candidate_cells.pop(0)
            
            # Try to create a continuous wall (1-3 cells long)
            obstacle_points = [(x, y)]
            maze[y][x] = 1  # Set the first obstacle
            obstacles_added += 1
            
            # Randomly choose a direction and try to extend the obstacle
            if neighbors and obstacles_added < num_obstacles:
                next_x, next_y = random.choice(neighbors)
                # Ensure this point is still available
                if maze[next_y][next_x] == 0 and (next_x, next_y) not in protected_cells:
                    obstacle_points.append((next_x, next_y))
                    maze[next_y][next_x] = 1
                    obstacles_added += 1
                    
                    # Potentially continue extending in the same direction
                    dx, dy = next_x - x, next_y - y
                    third_x, third_y = next_x + dx, next_y + dy
                    if (0 <= third_x < size and 0 <= third_y < size and 
                        maze[third_y][third_x] == 0 and 
                        (third_x, third_y) not in protected_cells and
                        obstacles_added < num_obstacles):
                        obstacle_points.append((third_x, third_y))
                        maze[third_y][third_x] = 1
                        obstacles_added += 1
            
            # Verify that a path still exists after adding the obstacles
            if not self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                # If there is no path, remove these obstacles
                for ox, oy in obstacle_points:
                    maze[oy][ox] = 0
                obstacles_added -= len(obstacle_points)
                continue
                
            # Update the list of candidate points, removing those that are no longer suitable
            candidate_cells = [cell for cell in candidate_cells 
                               if maze[cell[1]][cell[0]] == 0 and 
                               (cell[0], cell[1]) not in protected_cells]

    def is_wall(self, x, y):
        """Check if a position is a wall"""
        grid_x, grid_y = int(x), int(y)
        if 0 <= grid_x < self.MAZE_SIZE and 0 <= grid_y < self.MAZE_SIZE:
            return self.maze[grid_y][grid_x] == 1
        return True
    
    def cast_ray(self, angle):
        """Cast a ray from the player's position at a specific angle"""
        # Direction vector of the line of sight
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        
        # Player position
        pos_x, pos_y = self.player_pos
        map_x, map_y = int(pos_x), int(pos_y)
        
        # DDA algorithm parameters
        delta_dist_x = abs(1 / dir_x) if dir_x != 0 else float('inf')
        delta_dist_y = abs(1 / dir_y) if dir_y != 0 else float('inf')
        
        if dir_x < 0:
            step_x = -1
            side_dist_x = (pos_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - pos_x) * delta_dist_x
            
        if dir_y < 0:
            step_y = -1
            side_dist_y = (pos_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - pos_y) * delta_dist_y
            
        # Execute DDA algorithm
        hit = False
        side = 0  # 0 for x-side, 1 for y-side
        
        while not hit and (abs(map_x - pos_x) < self.MAX_DEPTH or abs(map_y - pos_y) < self.MAX_DEPTH):
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            
            if 0 <= map_x < self.MAZE_SIZE and 0 <= map_y < self.MAZE_SIZE:
                if self.maze[map_y][map_x] == 1:
                    hit = True
            else:
                hit = True
        
        # Calculate the exact distance
        if side == 0:
            wall_dist = (map_x - pos_x + (1 - step_x) / 2) / (dir_x if abs(dir_x) > 1e-6 else 1e-6)
        else:
            wall_dist = (map_y - pos_y + (1 - step_y) / 2) / (dir_y if abs(dir_y) > 1e-6 else 1e-6)
            
        # Ensure wall_dist is always positive and has a reasonable upper limit
        wall_dist = max(0.1, min(wall_dist, self.MAX_DEPTH))
        
        return wall_dist, side, (map_x, map_y)
    
    def render_3d_view(self):
        """Render the 3D view"""
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw sky and ground
        sky_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT//2)
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT//2, self.SCREEN_WIDTH, self.SCREEN_HEIGHT//2)
        pygame.draw.rect(self.screen, (100, 150, 200), sky_rect)
        pygame.draw.rect(self.screen, (50, 100, 50), ground_rect)
        
        # Save z-buffer for occlusion culling of the goal indicator
        z_buffer = [float('inf')] * self.SCREEN_WIDTH
        
        # Ray casting parameters
        ray_step = self.SCREEN_WIDTH / self.RAY_COUNT
        wall_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Cast rays
        for i in range(self.RAY_COUNT):
            x = int(i * ray_step)
            
            # Calculate ray direction
            camera_x = 2 * x / self.SCREEN_WIDTH - 1
            ray_angle = self.player_angle + camera_x * self.player_fov / 2
            
            # Cast the ray
            wall_dist, side, wall_pos = self.cast_ray(ray_angle)
            
            # Update the z-buffer
            end_x = int((i + 1) * ray_step) if i < self.RAY_COUNT - 1 else self.SCREEN_WIDTH
            for fill_x in range(x, end_x):
                z_buffer[min(fill_x, self.SCREEN_WIDTH - 1)] = wall_dist
            
            # Calculate wall height
            line_height = int(self.SCREEN_HEIGHT / wall_dist * self.WALL_HEIGHT)
            line_height = min(line_height, self.SCREEN_HEIGHT * 3)
                
            # Calculate drawing position
            draw_start = max(0, -line_height // 2 + self.SCREEN_HEIGHT // 2)
            draw_end = min(self.SCREEN_HEIGHT - 1, line_height // 2 + self.SCREEN_HEIGHT // 2)
            
            # Draw the wall with a solid color
            strip_width = int(ray_step) + 1
            
            # Choose color based on wall side
            base_color = (160, 160, 160) if side == 0 else (130, 130, 130)
            
            # Apply lighting
            if self.use_lighting:
                shade = max(0.3, min(1.0, 1.0 - wall_dist/self.MAX_DEPTH))
                color = tuple(int(c * shade) for c in base_color)
            else:
                color = base_color
            
            # Draw the wall
            pygame.draw.rect(wall_surface, color, (x, draw_start, strip_width, draw_end - draw_start))
        
        # Render the walls
        self.screen.blit(wall_surface, (0, 0))
        
        # Update the frame counter
        self.frame_counter += 1
        
        # Render the goal indicator
        self.render_goal_indicator(z_buffer)
        
        # Render path hints
        if self.show_path_hints:
            self.render_path_hints()
    
    def render_goal_indicator(self, z_buffer):
        """Render the goal indicator"""
        # Calculate goal direction
        goal_dir = self.goal_pos - self.player_pos
        goal_angle = math.atan2(goal_dir[1], goal_dir[0])
        
        # Calculate angle difference
        angle_diff = goal_angle - self.player_angle
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        # Calculate distance
        dist = np.linalg.norm(goal_dir)
        
        # If within the field of view and not occluded by a wall
        if abs(angle_diff) < self.player_fov / 2:
            # Calculate screen x position
            screen_x = int(self.SCREEN_WIDTH / 2 + (angle_diff / max(self.player_fov/2, 1e-6)) * (self.SCREEN_WIDTH / 2))
            screen_x = max(0, min(screen_x, self.SCREEN_WIDTH - 1))
            
            # Check for occlusion
            if dist >= z_buffer[screen_x]:
                return
            
            # Calculate icon size
            size = min(100, int(2000 / dist))
            pulse = (math.sin(self.frame_counter / 10) + 1) / 2
            size_with_pulse = int(size * (0.9 + 0.1 * pulse))
            
            # Adjust position to the ground
            ground_y_pos = int(self.SCREEN_HEIGHT * 0.75)
            height_adj = int((1.0 - min(1.0, dist / self.MAX_DEPTH)) * self.SCREEN_HEIGHT * 0.25)
            icon_y_pos = ground_y_pos - height_adj - size_with_pulse // 2
            
            # Draw the icon
            light_pillar_height = 0  # Default value for when goal_texture is not available
            if self.goal_texture:
                scaled_goal = pygame.transform.scale(self.goal_texture, (size_with_pulse, size_with_pulse))
                self.screen.blit(scaled_goal, (screen_x - size_with_pulse/2, icon_y_pos))
                
                # Add a glow effect
                glow_size = size_with_pulse + 10
                glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (0, 255, 0, 50), (glow_size//2, glow_size//2), glow_size//2)
                self.screen.blit(glow_surface, (screen_x - glow_size/2, icon_y_pos))
                
                # Draw a light pillar
                light_pillar_height = int(size_with_pulse * 0.7)
                light_pillar_surface = pygame.Surface((size_with_pulse//3, light_pillar_height), pygame.SRCALPHA)
                pillar_gradient = [(0, 255, 0, int(200 * (1 - i/light_pillar_height))) for i in range(light_pillar_height)]
                for i, color in enumerate(pillar_gradient):
                    pygame.draw.line(light_pillar_surface, color, (size_with_pulse//6, i), (size_with_pulse//6, i), size_with_pulse//3)
                self.screen.blit(light_pillar_surface, (screen_x - size_with_pulse//6, icon_y_pos + size_with_pulse))
            else:
                color = (0, int(200 + 55 * pulse), 0)
                pygame.draw.circle(self.screen, color, (screen_x, icon_y_pos + size_with_pulse/2), size_with_pulse/2)
            
            # Display distance
            font = pygame.font.SysFont(None, 24)
            dist_text = font.render(f"{dist:.1f}m", True, (255, 255, 255))
            text_y_pos = icon_y_pos + size_with_pulse + light_pillar_height + 5
            self.screen.blit(dist_text, (screen_x - dist_text.get_width()/2, text_y_pos))
    
    def render_path_hints(self):
        """Render path hints"""
        goal_dir = self.goal_pos - self.player_pos
        goal_dist = np.linalg.norm(goal_dir)
        
        if 2.0 < goal_dist < 10.0:
            # Arrow parameters
            arrow_color = (0, 220, 0, 100)
            arrow_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT * 3 // 4)
            arrow_size = 40
            
            # Calculate arrow angle
            goal_angle = math.atan2(goal_dir[1], goal_dir[0])
            relative_angle = goal_angle - self.player_angle
            while relative_angle > math.pi: relative_angle -= 2 * math.pi
            while relative_angle < -math.pi: relative_angle += 2 * math.pi
            
            # Calculate arrow endpoint
            end_x = arrow_pos[0] + int(math.cos(relative_angle) * arrow_size)
            end_y = arrow_pos[1] + int(math.sin(relative_angle) * arrow_size)
            
            # Calculate arrowhead
            head_angle1 = relative_angle + math.radians(150)
            head_angle2 = relative_angle + math.radians(210)
            head_len = 15
            head_x1 = end_x + int(math.cos(head_angle1) * head_len)
            head_y1 = end_y + int(math.sin(head_angle1) * head_len)
            head_x2 = end_x + int(math.cos(head_angle2) * head_len)
            head_y2 = end_y + int(math.sin(head_angle2) * head_len)
            
            # Draw the arrow
            pygame.draw.line(self.screen, arrow_color, arrow_pos, (end_x, end_y), 5)
            pygame.draw.line(self.screen, arrow_color, (end_x, end_y), (head_x1, head_y1), 5)
            pygame.draw.line(self.screen, arrow_color, (end_x, end_y), (head_x2, head_y2), 5)
    
    def render_mini_map(self):
        """Render the minimap"""
        if not self.show_minimap:
            return
        
        # Update the minimap every 3 frames
        map_size = 150
        cell_size = map_size // self.current_maze_size  # Use current maze size
        
        if self.minimap_surface is None or self.minimap_update_counter % 3 == 0:
            self.minimap_surface = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
            self.minimap_surface.fill((0, 0, 0, 180))
            
            # Draw walls
            for y in range(self.current_maze_size):
                for x in range(self.current_maze_size):
                    if x < len(self.maze[0]) and y < len(self.maze) and self.maze[y][x] == 1:
                        color = (150, 150, 200, 200) if x == 0 or y == 0 or x == self.current_maze_size-1 or y == self.current_maze_size-1 else (200, 200, 200, 200)
                        pygame.draw.rect(self.minimap_surface, color, (x * cell_size, y * cell_size, cell_size, cell_size))
        
        # Draw player and goal
        player_layer = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
        player_layer.fill((0, 0, 0, 0))
        
        # Draw player position
        px = int(self.player_pos[0] * cell_size)
        py = int(self.player_pos[1] * cell_size)
        
        # Draw field of view cone
        left_angle = self.player_angle - self.player_fov/2
        right_angle = self.player_angle + self.player_fov/2
        left_dx = math.cos(left_angle) * 30
        left_dy = math.sin(left_angle) * 30
        right_dx = math.cos(right_angle) * 30
        right_dy = math.sin(right_angle) * 30
        
        points = [(px, py), (px + int(left_dx), py + int(left_dy)), (px + int(right_dx), py + int(right_dy))]
        pygame.draw.polygon(player_layer, (255, 255, 0, 40), points)
        
        # Draw player
        pygame.draw.circle(player_layer, (255, 50, 50, 200), (px, py), cell_size//2)
        
        # Draw direction indicator
        end_x = px + int(cell_size * math.cos(self.player_angle))
        end_y = py + int(cell_size * math.sin(self.player_angle))
        pygame.draw.line(player_layer, (255, 0, 0, 200), (px, py), (end_x, end_y), 2)
        
        # Draw goal
        gx = int(self.goal_pos[0] * cell_size)
        gy = int(self.goal_pos[1] * cell_size)
        pulse = (math.sin(self.frame_counter / 10) + 1) / 2
        goal_radius = int(cell_size * (1.0 + 0.3 * pulse))
        
        pygame.draw.circle(player_layer, (0, 255, 0, 50), (gx, gy), goal_radius + 6)
        pygame.draw.circle(player_layer, (0, 255, 0, 100), (gx, gy), goal_radius + 3)
        pygame.draw.circle(player_layer, (0, 255, 0, 200), (gx, gy), goal_radius)
        
        # Draw the minimap
        combined_map = self.minimap_surface.copy()
        combined_map.blit(player_layer, (0, 0))
        
        border = pygame.Surface((map_size + 4, map_size + 4), pygame.SRCALPHA)
        border.fill((255, 255, 255, 100))
        self.screen.blit(border, (self.SCREEN_WIDTH - map_size - 14, 8))
        self.screen.blit(combined_map, (self.SCREEN_WIDTH - map_size - 12, 10))
        
        self.minimap_update_counter += 1
    
    def update_fps(self):
        """Calculate and display FPS"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fps_time > 1000:
            self.current_fps = len(self.fps_values)
            self.fps_values = []
            self.last_fps_time = current_time
        else:
            self.fps_values.append(1)
        
        font = pygame.font.SysFont(None, 24)
        fps_text = font.render(f"FPS: {self.current_fps}", True, (255, 255, 255))
        self.screen.blit(fps_text, (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 30))
    
    def handle_input(self):
        """Handle user input"""
        keys = pygame.key.get_pressed()
        
        # Rotation
        if keys[K_LEFT] or keys[K_a]:
            self.player_angle -= self.TURN_SPEED
                
        if keys[K_RIGHT] or keys[K_d]:
            self.player_angle += self.TURN_SPEED

        
        # Movement calculation
        dir_x, dir_y = math.cos(self.player_angle), math.sin(self.player_angle)
        right_x = math.cos(self.player_angle + math.pi/2)
        right_y = math.sin(self.player_angle + math.pi/2)
        
        # Forward/backward movement
        if keys[K_UP] or keys[K_w]:
            new_x = self.player_pos[0] + dir_x * self.MOVE_SPEED
            new_y = self.player_pos[1] + dir_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
                
        if keys[K_DOWN] or keys[K_s]:
            new_x = self.player_pos[0] - dir_x * self.MOVE_SPEED
            new_y = self.player_pos[1] - dir_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
        
        # Strafing
        if keys[K_q]:
            new_x = self.player_pos[0] - right_x * self.MOVE_SPEED
            new_y = self.player_pos[1] - right_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
                
        if keys[K_e]:
            new_x = self.player_pos[0] + right_x * self.MOVE_SPEED
            new_y = self.player_pos[1] + right_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
        
        # Toggle options
        for event in pygame.event.get(KEYDOWN):
            if event.key == K_l:
                self.use_lighting = not self.use_lighting
            elif event.key == K_m:
                self.show_minimap = not self.show_minimap
            elif event.key == K_h:
                self.show_path_hints = not self.show_path_hints
    
    def check_goal(self):
        """Check if the goal is reached"""
        dist_to_goal = np.linalg.norm(self.player_pos - self.goal_pos)
        # Set different goal achievement distances based on difficulty
        threshold = 1.0 if self.difficulty == 'hard' else 0.7
        if dist_to_goal < threshold:
            self.won = True
    
    def render_ui(self):
        """Render UI elements"""
        # Display position information
        font = pygame.font.SysFont(None, 24)
        pos_text = font.render(f"Position: ({self.player_pos[0]:.1f}, {self.player_pos[1]:.1f})", True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 10))
        
        # If won, display information
        if self.won:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            large_font = pygame.font.SysFont(None, 64)
            win_text = large_font.render("Congratulations! Exit found!", True, (255, 255, 0))
            text_rect = win_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 50))
            self.screen.blit(win_text, text_rect)
            
            if (self.frame_counter // 30) % 2 == 0:
                restart_text = font.render("Press R to restart or ESC to exit", True, (255, 255, 255))
                restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 + 50))
                self.screen.blit(restart_text, restart_rect)
        
        # Display FPS
        self.update_fps()
    
    def reset_game(self, difficulty=None):
        """Reset the game, can optionally change difficulty"""
        if difficulty and difficulty in self.DIFFICULTY_SETTINGS:
            self.difficulty = difficulty
            self.difficulty_config = self.DIFFICULTY_SETTINGS[difficulty]
            pygame.display.set_caption(f"3D Maze Adventure - {self.difficulty.capitalize()}")
            
            # Update maze size
            self.current_maze_size = self.MAZE_SIZE + self.difficulty_config['maze_size_increase']
            # Update exit position
            self.goal_pos = np.array([self.current_maze_size - 1.5, self.current_maze_size - 1.5])
            
        self.maze = self.generate_simplified_maze(self.current_maze_size)
        self.player_pos = np.array([1.5, 1.5])
        self.player_angle = 0
        self.won = False
        if self.sound_go:
            self.sound_go.play()
    
    def run(self):
        """Main game loop"""
        if self.sound_go:
            self.sound_go.play()
        
        while self.running:
            # Event handling
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_r:
                        self.reset_game()
            
            # Game logic
            if not self.won:
                self.handle_input()
                self.check_goal()
            
            # Rendering
            self.render_3d_view()
            self.render_mini_map()
            self.render_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

    def create_path_to_exit(self, maze, start_x, start_y, end_x, end_y):
        """Create a path from the start to the exit"""
        # Use a simple method to create the path - first move horizontally, then vertically
        current_x, current_y = start_x, start_y
        
        # First, move horizontally to the same x-coordinate as the end point
        while current_x < end_x:
            current_x += 1
            maze[current_y][current_x] = 0  # Set as path
    
        # Then, move vertically to the end point
        while current_y < end_y:
            current_y += 1
            maze[current_y][current_x] = 0  # Set as path
    
        return maze

    def find_shortest_path(self, maze, start, end):
        """Use BFS algorithm to find the shortest path from start to end"""
        queue = [(start, [start])]  # Element is (current_position, path_from_start_to_current)
        visited = set([start])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Four directions: down, right, up, left
        size = len(maze)
        
        while queue:
            (x, y), path = queue.pop(0)  # Get the first element of the queue
            
            # If the end is reached
            if (x, y) == end:
                return path
                
            # Explore the four directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # If the new position is valid, a path, and not visited
                if (0 <= nx < size and 0 <= ny < size and 
                    maze[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
        
        return None  # If no path is found

    def narrow_paths(self, maze, size):
        """Adjust path width based on difficulty"""
        narrow_factor = self.difficulty_config.get('narrow_path_factor', 0.0)
        if narrow_factor <= 0:
            return
            
        # Create an obstacle marker array
        obstacles = np.zeros((size, size), dtype=np.int8)
        
        # For each path cell in the maze
        for y in range(1, size-1):
            for x in range(1, size-1):
                if maze[y][x] == 0:  # If it's a path
                    # Check if this cell can become an obstacle (narrow the path)
                    if random.random() < narrow_factor:
                        # Temporarily mark this cell as a wall
                        maze[y][x] = 1
                        
                        # Check if this change would block the path
                        if self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                            # If it doesn't block, keep the mark
                            obstacles[y][x] = 1
                        else:
                            # Otherwise, restore it as a path
                            maze[y][x] = 0
                
        # Apply the final obstacle markings
        for y in range(size):
            for x in range(size):
                if obstacles[y][x] == 1:
                    maze[y][x] = 1
        
        return maze
    
    def is_path_exists(self, maze, start, end):
        """
        Check if a path exists from start to end in the maze
        
        Parameters:
            maze: The maze matrix
            start: Start coordinates (x, y)
            end: End coordinates (x, y)
            
        Returns:
            bool: Returns True if a path exists, otherwise False
        """
        # Use Breadth-First Search algorithm
        queue = [start]
        visited = set([start])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
        
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
                
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and
                    maze[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        
        return False