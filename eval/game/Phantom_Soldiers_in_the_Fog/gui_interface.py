#!/usr/bin/env python3
"""
Pygame GUI Interface for Cooperative Command Game

Provides a visual interface showing the map, team members, objectives, and game status.
Integrates with the gym environment for LLM evaluation and RL training.
"""

import pygame
import math
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Try relative imports first, then absolute
try:
    from .env import CoopCommandEnv, GameConfig, GameDifficulty
    from .visualization import GameVisualization, MapTileType
except ImportError:
    from env import CoopCommandEnv, GameConfig, GameDifficulty
    from visualization import GameVisualization, MapTileType


@dataclass
class GUIConfig:
    """Configuration for the GUI"""
    window_width: int = 1200
    window_height: int = 800
    map_width: int = 800
    map_height: int = 600
    panel_width: int = 400
    tile_size: int = 8
    fps: int = 60


class Colors:
    """Color constants for the GUI"""
    # Map colors
    OPEN = (144, 238, 144)      # Light green
    COVER = (139, 69, 19)       # Brown
    BUILDING = (105, 105, 105)  # Gray
    WATER = (65, 105, 225)      # Blue
    OBJECTIVE = (255, 215, 0)   # Gold
    
    # UI colors
    BACKGROUND = (30, 30, 30)
    PANEL_BG = (50, 50, 50)
    TEXT = (255, 255, 255)
    TEXT_HIGHLIGHT = (255, 255, 0)
    BORDER = (100, 100, 100)
    
    # Team member colors
    SCOUT = (0, 255, 0)         # Green
    HEAVY = (255, 0, 0)         # Red
    MEDIC = (255, 255, 255)     # White
    ENGINEER = (255, 255, 0)    # Yellow
    SNIPER = (128, 0, 128)      # Purple
    SUPPORT = (255, 165, 0)     # Orange
    
    # Objective status colors
    PENDING = (255, 255, 0)     # Yellow
    IN_PROGRESS = (255, 165, 0) # Orange
    COMPLETED = (0, 255, 0)     # Green
    FAILED = (255, 0, 0)        # Red
    HIDDEN = (128, 128, 128)    # Gray


class GameGUI:
    """Main GUI class for the cooperative command game"""
    
    def __init__(self, config: GUIConfig = None):
        self.config = config or GUIConfig()
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Create window
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("Cooperative Command Game")
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Game components
        self.env = None
        self.visualization = GameVisualization()
        
        # GUI state
        self.running = True
        self.game_started = False
        self.current_state = None
        
        # Map rendering
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.zoom_level = 1.0
        
        # Panels
        self.map_rect = pygame.Rect(0, 0, self.config.map_width, self.config.map_height)
        self.info_panel_rect = pygame.Rect(self.config.map_width, 0, self.config.panel_width, self.config.window_height)

    def create_game(self, difficulty: str = "normal", seed_index: int = 0, enable_audio: bool = False):
        """Create a new game environment"""
        try:
            config = GameConfig(
                difficulty=GameDifficulty[difficulty.upper()],
                seed_index=seed_index,
                enable_audio=enable_audio
            )
            self.env = CoopCommandEnv(config=config, enable_assets=False)
            self.current_state = self.env.start_game()
            self.game_started = True
            print(f"Game created: {difficulty} difficulty, seed {seed_index}")
        except Exception as e:
            print(f"Error creating game: {e}")
            self.game_started = False

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and not self.game_started:
                    # Create a random game
                    self.create_game()
                elif event.key == pygame.K_SPACE and self.game_started:
                    # Execute a random command for demo
                    self._execute_demo_command()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos, event.button)
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                if event.y > 0:
                    self.zoom_level = min(3.0, self.zoom_level * 1.1)
                else:
                    self.zoom_level = max(0.5, self.zoom_level * 0.9)

    def _handle_mouse_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks"""
        mouse_x, mouse_y = pos
        
        # Check if click is on map
        if self.map_rect.collidepoint(mouse_x, mouse_y):
            # Convert screen coordinates to world coordinates
            world_x, world_y = self._screen_to_world(mouse_x, mouse_y)
            print(f"Clicked on map at world coordinates: ({world_x:.1f}, {world_y:.1f})")
            
            if self.game_started and button == 1:  # Left click
                # Demo: move first team member to clicked location
                self._move_member_to_position(0, world_x, world_y)

    def _execute_demo_command(self):
        """Execute a demo command for testing"""
        if not self.game_started or not self.env:
            return
        
        import random
        commands = [
            "move to position 25, 30",
            "scout area 40, 20", 
            "report status",
            "move to position 60, 45"
        ]
        
        command = random.choice(commands)
        result = self.env.execute_command(command)
        self.current_state = self.env.step()
        print(f"Executed: {command} -> {result['response']}")

    def _move_member_to_position(self, member_index: int, x: float, y: float):
        """Move a team member to the specified position"""
        if not self.game_started or not self.env:
            return
        
        # Get team member IDs
        members = list(self.current_state['team_status']['members'].keys())
        if member_index < len(members):
            command = f"move to position {int(x)}, {int(y)}"
            result = self.env.execute_command(command)
            self.current_state = self.env.step()
            print(f"Moving {members[member_index]} to ({x:.1f}, {y:.1f})")

    def _world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Scale world coordinates (0-100) to map dimensions (800x600)
        # Add padding to keep objectives away from edges
        padding = 20
        screen_x = int(padding + (world_x / 100.0) * (self.config.map_width - 2 * padding) * self.zoom_level + self.map_offset_x)
        screen_y = int(padding + (world_y / 100.0) * (self.config.map_height - 2 * padding) * self.zoom_level + self.map_offset_y)
        return screen_x, screen_y

    def _screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        # Convert back from screen to world coordinates
        padding = 20
        world_x = ((screen_x - self.map_offset_x) / self.zoom_level - padding) * 100.0 / (self.config.map_width - 2 * padding)
        world_y = ((screen_y - self.map_offset_y) / self.zoom_level - padding) * 100.0 / (self.config.map_height - 2 * padding)
        return world_x, world_y

    def render(self):
        """Render the entire GUI"""
        # Clear screen
        self.screen.fill(Colors.BACKGROUND)
        
        if self.game_started and self.current_state:
            # Get render data from visualization system
            render_data = self.visualization.get_render_data(
                self.current_state['team_status'],
                self.current_state.get('hidden_objectives', [])
            )
            
            # Render map
            self._render_map(render_data)
            
            # Render info panel
            self._render_info_panel(render_data)
        else:
            # Render start screen
            self._render_start_screen()
        
        # Update display
        pygame.display.flip()

    def _render_map(self, render_data: Dict[str, Any]):
        """Render the game map"""
        # Create map surface
        map_surface = pygame.Surface((self.config.map_width, self.config.map_height))
        map_surface.fill(Colors.OPEN)  # Default background
        
        # Render tiles (simplified - just background for now)
        tile_size = int(self.config.tile_size * self.zoom_level)
        
        # Render grid (optional)
        if self.zoom_level > 1.5:
            for x in range(0, self.config.map_width, tile_size):
                pygame.draw.line(map_surface, Colors.BORDER, (x, 0), (x, self.config.map_height))
            for y in range(0, self.config.map_height, tile_size):
                pygame.draw.line(map_surface, Colors.BORDER, (0, y), (self.config.map_width, y))
        
        # Render objectives
        for obj in render_data['objectives']:
            self._render_objective(map_surface, obj)
        
        # Render team members
        for member in render_data['team_members']:
            self._render_team_member(map_surface, member)
        
        # Blit map to main screen
        self.screen.blit(map_surface, self.map_rect)
        
        # Draw map border
        pygame.draw.rect(self.screen, Colors.BORDER, self.map_rect, 2)

    def _render_objective(self, surface: pygame.Surface, obj: Dict[str, Any]):
        """Render an objective on the map"""
        pos = obj['position']
        screen_x, screen_y = self._world_to_screen(pos['x'], pos['y'])
        
        # Skip if outside visible area
        if not (0 <= screen_x <= self.config.map_width and 0 <= screen_y <= self.config.map_height):
            return
        
        # Choose color based on status
        status = obj['status']
        if status == 'completed':
            color = Colors.COMPLETED
        elif status == 'in_progress':
            color = Colors.IN_PROGRESS
        elif status == 'failed':
            color = Colors.FAILED
        elif status == 'hidden':
            color = Colors.HIDDEN
        else:
            color = Colors.PENDING
        
        # Draw objective marker
        radius = max(3, int(6 * self.zoom_level))
        pygame.draw.circle(surface, color, (screen_x, screen_y), radius)
        pygame.draw.circle(surface, Colors.TEXT, (screen_x, screen_y), radius, 2)
        
        # Draw objective type text if zoomed in
        if self.zoom_level > 1.0:
            obj_type = obj.get('type', 'obj')
            text_surface = self.font_small.render(obj_type, True, Colors.TEXT)
            text_rect = text_surface.get_rect(center=(screen_x, screen_y - radius - 10))
            surface.blit(text_surface, text_rect)

    def _render_team_member(self, surface: pygame.Surface, member: Dict[str, Any]):
        """Render a team member on the map"""
        pos = member['position']
        screen_x, screen_y = self._world_to_screen(pos['x'], pos['y'])
        
        # Skip if outside visible area
        if not (0 <= screen_x <= self.config.map_width and 0 <= screen_y <= self.config.map_height):
            return
        
        # Choose color based on role
        role = member['role'].lower()
        color_map = {
            'scout': Colors.SCOUT,
            'heavy': Colors.HEAVY,
            'medic': Colors.MEDIC,
            'engineer': Colors.ENGINEER,
            'sniper': Colors.SNIPER,
            'support': Colors.SUPPORT
        }
        color = color_map.get(role, Colors.TEXT)
        
        # Draw team member
        radius = max(4, int(8 * self.zoom_level))
        pygame.draw.circle(surface, color, (screen_x, screen_y), radius)
        pygame.draw.circle(surface, Colors.TEXT, (screen_x, screen_y), radius, 2)
        
        # Draw health bar if zoomed in
        if self.zoom_level > 1.0:
            health = member['health']
            bar_width = int(20 * self.zoom_level)
            bar_height = 4
            bar_x = screen_x - bar_width // 2
            bar_y = screen_y + radius + 2
            
            # Background
            pygame.draw.rect(surface, Colors.BACKGROUND, (bar_x, bar_y, bar_width, bar_height))
            # Health
            health_width = int(bar_width * health / 100)
            if health > 60:
                health_color = Colors.COMPLETED
            elif health > 30:
                health_color = Colors.IN_PROGRESS
            else:
                health_color = Colors.FAILED
            pygame.draw.rect(surface, health_color, (bar_x, bar_y, health_width, bar_height))
        
        # Draw member name if zoomed in enough
        if self.zoom_level > 1.5:
            name = member.get('name', member['id'])
            text_surface = self.font_small.render(name, True, Colors.TEXT)
            text_rect = text_surface.get_rect(center=(screen_x, screen_y - radius - 10))
            surface.blit(text_surface, text_rect)

    def _render_info_panel(self, render_data: Dict[str, Any]):
        """Render the information panel with height management"""
        # Fill panel background
        pygame.draw.rect(self.screen, Colors.PANEL_BG, self.info_panel_rect)
        pygame.draw.rect(self.screen, Colors.BORDER, self.info_panel_rect, 2)
        
        y_offset = 10
        x_start = self.info_panel_rect.x + 10
        panel_bottom = self.info_panel_rect.bottom - 20  # Reserve space at bottom
        
        # Game status
        title = self.font_large.render("Game Status", True, Colors.TEXT_HIGHLIGHT)
        self.screen.blit(title, (x_start, y_offset))
        y_offset += 30
        
        # Score information
        score_text = f"Score: {self.current_state.get('main_score', 0)}"
        score_surface = self.font_medium.render(score_text, True, Colors.TEXT)
        self.screen.blit(score_surface, (x_start, y_offset))
        y_offset += 20
        
        normalized_score = self.current_state.get('score_normalized', 0)
        norm_text = f"Normalized: {normalized_score:.1f}/100"
        norm_surface = self.font_medium.render(norm_text, True, Colors.TEXT)
        self.screen.blit(norm_surface, (x_start, y_offset))
        y_offset += 30
        
        # Round information
        round_text = f"Round: {self.current_state.get('current_round', 0)}/{self.current_state.get('max_rounds', 0)}"
        round_surface = self.font_medium.render(round_text, True, Colors.TEXT)
        self.screen.blit(round_surface, (x_start, y_offset))
        y_offset += 30
        
        # Team members - with height checking
        team_title = self.font_large.render("Team Members", True, Colors.TEXT_HIGHLIGHT)
        self.screen.blit(team_title, (x_start, y_offset))
        y_offset += 25
        
        # Calculate space needed for team members and adjust if necessary
        team_members = render_data['team_members']
        team_height_per_member = 53  # 18 + 15 + 20 for name, pos, health
        team_total_height = len(team_members) * team_height_per_member
        
        # If too many team members, use compact display
        if y_offset + team_total_height > panel_bottom - 200:  # Reserve 200px for objectives
            # Compact team display
            for member in team_members:
                if y_offset + 20 > panel_bottom - 180:
                    break  # Stop if running out of space
                # Single line: Name (Role) - Health% at (x,y)
                compact_text = f"{member['name']} ({member['role']}) - {member['health']}% at ({member['position']['x']:.0f},{member['position']['y']:.0f})"
                if len(compact_text) > 40:
                    compact_text = compact_text[:37] + "..."
                compact_surface = self.font_small.render(compact_text, True, Colors.TEXT)
                self.screen.blit(compact_surface, (x_start, y_offset))
                y_offset += 16
        else:
            # Normal team display
            for member in team_members:
                if y_offset + 53 > panel_bottom - 150:
                    break  # Stop if running out of space
                # Member name and role
                name = f"{member['name']} ({member['role']})"
                name_surface = self.font_medium.render(name, True, Colors.TEXT)
                self.screen.blit(name_surface, (x_start, y_offset))
                y_offset += 18
                
                # Position and health
                pos_text = f"  Pos: ({member['position']['x']:.1f}, {member['position']['y']:.1f})"
                pos_surface = self.font_small.render(pos_text, True, Colors.TEXT)
                self.screen.blit(pos_surface, (x_start, y_offset))
                y_offset += 15
                
                health_text = f"  Health: {member['health']}% Status: {member['status']}"
                health_surface = self.font_small.render(health_text, True, Colors.TEXT)
                self.screen.blit(health_surface, (x_start, y_offset))
                y_offset += 20
        
        # Objectives section - with height management
        total_objectives = self.current_state.get('total_objectives_count', len(render_data['objectives']))
        hidden_count = self.current_state.get('hidden_objectives_count', 0)
        visible_count = len(render_data['objectives'])
        
        # Check remaining space for objectives
        remaining_height = panel_bottom - y_offset - 50  # Reserve 50px buffer
        
        obj_title = f"Objectives ({visible_count}/{total_objectives} visible)"
        obj_title_surface = self.font_large.render(obj_title, True, Colors.TEXT_HIGHLIGHT)
        self.screen.blit(obj_title_surface, (x_start, y_offset))
        y_offset += 25
        
        # Show hidden objectives info
        if hidden_count > 0:
            hidden_text = f"🔍 {hidden_count} hidden objectives - explore to discover!"
            hidden_surface = self.font_small.render(hidden_text, True, Colors.HIDDEN)
            self.screen.blit(hidden_surface, (x_start, y_offset))
            y_offset += 18
            remaining_height -= 18
        
        # Show visible objectives with height constraints
        objectives = render_data['objectives']
        objectives_shown = 0
        estimated_height_per_obj = 35  # Updated estimate for two-line objectives
        max_objectives_that_fit = max(2, remaining_height // estimated_height_per_obj)  # Show at least 2
        
        for obj in objectives:
            if objectives_shown >= max_objectives_that_fit:
                # Show truncation indicator
                if len(objectives) > objectives_shown:
                    truncate_text = f"... and {len(objectives) - objectives_shown} more (scroll needed)"
                    truncate_surface = self.font_small.render(truncate_text, True, Colors.HIDDEN)
                    self.screen.blit(truncate_surface, (x_start, y_offset))
                break
            
            # Check if we have enough space for this objective (two lines)
            if y_offset + estimated_height_per_obj > panel_bottom:
                break
            
            # Status indicator
            status = obj['status']
            status_colors = {
                'completed': Colors.COMPLETED,
                'in_progress': Colors.IN_PROGRESS,
                'failed': Colors.FAILED,
                'hidden': Colors.HIDDEN,
                'pending': Colors.PENDING
            }
            status_color = status_colors.get(status, Colors.TEXT)
            
            # Status circle
            pygame.draw.circle(self.screen, status_color, (x_start + 5, y_offset + 8), 4)
            
            # Objective text - smart wrapping for full visibility
            obj_desc = obj.get('description', obj.get('type', 'objective'))
            
            # Smart text wrapping to ensure coordinates are always visible
            max_line_length = 38  # Characters that fit in the panel width
            
            if len(obj_desc) <= max_line_length:
                # Single line - fits completely
                obj_surface = self.font_small.render(obj_desc, True, Colors.TEXT)
                self.screen.blit(obj_surface, (x_start + 15, y_offset))
                y_offset += 18
            else:
                # Two lines needed - smart split to preserve coordinates
                if " at (" in obj_desc:
                    # Split at coordinate boundary to preserve full coordinates
                    parts = obj_desc.split(" at (", 1)
                    main_part = parts[0].strip()
                    coord_part = "at (" + parts[1] if len(parts) > 1 else ""
                    
                    # If main part is too long, break it intelligently
                    if len(main_part) > max_line_length:
                        # Find a good break point (space, comma, etc.)
                        break_point = max_line_length - 3
                        for i in range(break_point, max(0, break_point - 10), -1):
                            if main_part[i] in [' ', ',', '-', '_']:
                                break_point = i
                                break
                        
                        line1 = main_part[:break_point].strip()
                        line2 = main_part[break_point:].strip()
                        if coord_part:
                            line2 += " " + coord_part
                    else:
                        # Main part fits on first line
                        line1 = main_part
                        line2 = coord_part
                    
                    # Render first line
                    line1_surface = self.font_small.render(line1, True, Colors.TEXT)
                    self.screen.blit(line1_surface, (x_start + 15, y_offset))
                    y_offset += 16
                    
                    # Render second line with slight indent and smaller font for coordinates
                    if line2:
                        line2_surface = pygame.font.Font(None, 14).render(line2, True, Colors.TEXT)
                        self.screen.blit(line2_surface, (x_start + 20, y_offset))
                        y_offset += 16
                    else:
                        y_offset += 2  # Small gap if no second line
                else:
                    # No coordinates - break at word boundary
                    break_point = max_line_length - 3
                    for i in range(break_point, max(0, break_point - 10), -1):
                        if obj_desc[i] in [' ', ',', '-', '_']:
                            break_point = i
                            break
                    
                    line1 = obj_desc[:break_point].strip()
                    line2 = obj_desc[break_point:].strip()
                    
                    # Render first line
                    line1_surface = self.font_small.render(line1, True, Colors.TEXT)
                    self.screen.blit(line1_surface, (x_start + 15, y_offset))
                    y_offset += 16
                    
                    # Render second line with slight indent
                    if line2:
                        line2_surface = self.font_small.render(line2, True, Colors.TEXT)
                        self.screen.blit(line2_surface, (x_start + 20, y_offset))
                        y_offset += 16
                    else:
                        y_offset += 2
            
            # Add small gap between objectives
            y_offset += 3
            objectives_shown += 1
        
        # Controls section - only if space allows
        if y_offset + 80 <= panel_bottom:  # Need at least 80px for controls
            y_offset += 10
            controls_title = self.font_large.render("Controls", True, Colors.TEXT_HIGHLIGHT)
            self.screen.blit(controls_title, (x_start, y_offset))
            y_offset += 25
            
            # Compact controls list
            controls = [
                "R - New game",
                "SPACE - Demo",
                "Click - Move"
            ]
            
            for control in controls:
                if y_offset + 15 <= panel_bottom:
                    control_surface = self.font_small.render(control, True, Colors.TEXT)
                    self.screen.blit(control_surface, (x_start, y_offset))
                    y_offset += 15

    def _render_start_screen(self):
        """Render the start screen"""
        # Title
        title = self.font_large.render("Cooperative Command Game", True, Colors.TEXT_HIGHLIGHT)
        title_rect = title.get_rect(center=(self.config.window_width // 2, self.config.window_height // 2 - 100))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Press R to start a new game",
            "Use mouse to interact with the map",
            "View team members and objectives in the right panel"
        ]
        
        y_offset = self.config.window_height // 2 - 50
        for instruction in instructions:
            text = self.font_medium.render(instruction, True, Colors.TEXT)
            text_rect = text.get_rect(center=(self.config.window_width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(self.config.fps)
        
        # Cleanup
        if self.env:
            self.env.shutdown()
        pygame.quit()


def main():
    """Main function to run the GUI"""
    gui = GameGUI()
    
    # Create a demo game
    gui.create_game(difficulty="normal", seed_index=0, enable_audio=False)
    
    # Run the GUI
    gui.run()


if __name__ == "__main__":
    main() 