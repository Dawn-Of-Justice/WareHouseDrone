import pygame
import sys
import math
import random  # Added for random enemy spawning
import numpy as np

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FOV = math.pi / 3  # Field of view
HALF_FOV = FOV / 2
MAX_DEPTH = 20
CELL_SIZE = 64
PLAYER_SPEED = 5
ROTATION_SPEED = 0.1
ENEMY_SPEED = 0.01  # Slow enemy movement speed
RAY_COUNT = SCREEN_WIDTH  # Number of rays to cast (one per screen column)
MIN_WALL_DISTANCE = 0.1 * CELL_SIZE  # Minimum distance to prevent rendering artifacts
RELOAD_TIME = 60  # Frames to reload (1 second at 60 FPS)
SPAWN_RATE = 300  # Frames between enemy spawns (5 seconds at 60 FPS)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Doom Clone")
clock = pygame.time.Clock()

# Disable mouse movement tracking initially to avoid abrupt camera movement
pygame.mouse.set_pos(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
pygame.mouse.get_rel()  # Clear initial relative movement

# Map (1 represents walls, 0 represents empty space)
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

MAP_WIDTH = len(MAP[0])
MAP_HEIGHT = len(MAP)

# Player starting position and direction
player_x = 3.5 * CELL_SIZE
player_y = 2.5 * CELL_SIZE
player_angle = 0

# Enemy positions with enhanced properties
enemies = [
    {"x": 5.5 * CELL_SIZE, "y": 5.5 * CELL_SIZE, "alive": True, "type": 0, "health": 1, "animation_frame": 0},
    {"x": 10.5 * CELL_SIZE, "y": 5.5 * CELL_SIZE, "alive": True, "type": 1, "health": 2, "animation_frame": 1},
    {"x": 15.5 * CELL_SIZE, "y": 7.5 * CELL_SIZE, "alive": True, "type": 2, "health": 3, "animation_frame": 2},
    {"x": 8.5 * CELL_SIZE, "y": 13.5 * CELL_SIZE, "alive": True, "type": 0, "health": 1, "animation_frame": 3},
    {"x": 12.5 * CELL_SIZE, "y": 16.5 * CELL_SIZE, "alive": True, "type": 1, "health": 2, "animation_frame": 0}
]

# Game state
bullets = []  # List to store active bullets
score = 0
health = 100
ammo = 100
max_ammo = 100
reloading = False
reload_timer = 0
game_over = False
spawn_timer = SPAWN_RATE  # Timer for enemy spawning
enemy_types = [
    {"color": RED, "health": 1, "speed": 0.01, "size": 1.0, "damage": 1, "points": 100},
    {"color": (255, 165, 0), "health": 2, "speed": 0.008, "size": 1.2, "damage": 2, "points": 200},
    {"color": (128, 0, 128), "health": 3, "speed": 0.006, "size": 1.5, "damage": 3, "points": 300}
]

# Textures - using simple color patterns for walls
def get_wall_color(ray_angle, wall_distance):
    # Ensure minimum distance to prevent color artifacts up close
    wall_distance = max(wall_distance, MIN_WALL_DISTANCE)
    
    # Darken walls based on distance with smoother falloff
    darkness = 1.0 - min(0.9, wall_distance / MAX_DEPTH)  # Cap at 0.9 to keep some color visible at distance
    intensity = int(255 * darkness)
    
    # Different colors for different wall orientations
    angle_mod = ray_angle % (math.pi * 2)
    if angle_mod < math.pi / 4 or angle_mod > 7 * math.pi / 4:  # East walls
        return (intensity, intensity // 2, 0)  # Orange walls
    elif angle_mod < 3 * math.pi / 4:  # North walls
        return (0, 0, intensity)  # Blue walls
    elif angle_mod < 5 * math.pi / 4:  # West walls
        return (intensity, 0, 0)  # Red walls
    else:  # South walls
        return (0, intensity, 0)  # Green walls

# Raycasting function with improved wall detection
def cast_rays():
    walls = []
    ray_angle = player_angle - HALF_FOV
    
    # Cast rays from player position
    for ray in range(SCREEN_WIDTH):
        # Calculate ray direction
        ray_dx = math.cos(ray_angle)
        ray_dy = math.sin(ray_angle)
        
        # Digital Differential Analysis (DDA) algorithm for raycasting
        # This is more efficient and accurate than step-by-step ray marching
        
        # Current position in map grid
        map_x = int(player_x / CELL_SIZE)
        map_y = int(player_y / CELL_SIZE)
        
        # Length of ray from current position to next x or y-side
        delta_dist_x = abs(1 / ray_dx) if ray_dx != 0 else float('inf')
        delta_dist_y = abs(1 / ray_dy) if ray_dy != 0 else float('inf')
        
        # Direction to step in x or y direction (either +1 or -1)
        step_x = 1 if ray_dx >= 0 else -1
        step_y = 1 if ray_dy >= 0 else -1
        
        # Length of ray from one x or y-side to next x or y-side
        # Starting position is current position
        ray_pos_x = player_x / CELL_SIZE - map_x
        ray_pos_y = player_y / CELL_SIZE - map_y
        
        # Calculate distance to first x and y intersections
        if ray_dx < 0:
            side_dist_x = ray_pos_x * delta_dist_x
        else:
            side_dist_x = (1.0 - ray_pos_x) * delta_dist_x
            
        if ray_dy < 0:
            side_dist_y = ray_pos_y * delta_dist_y
        else:
            side_dist_y = (1.0 - ray_pos_y) * delta_dist_y
        
        # Perform DDA until wall hit
        hit_wall = False
        side = 0  # 0 for x-side, 1 for y-side
        distance = 0
        
        while not hit_wall and distance < MAX_DEPTH:
            # Jump to next map square, either in x-direction, or in y-direction
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            
            # Check if ray has hit a wall
            if map_x < 0 or map_x >= MAP_WIDTH or map_y < 0 or map_y >= MAP_HEIGHT:
                hit_wall = True
                distance = MAX_DEPTH  # Set to max to show map boundary
            elif MAP[map_y][map_x] == 1:
                hit_wall = True
        
        # Calculate distance (avoiding fish-eye effect by using perpendicular distance)
        if side == 0:
            perp_wall_dist = (map_x - player_x / CELL_SIZE + (1 - step_x) / 2) / ray_dx
        else:
            perp_wall_dist = (map_y - player_y / CELL_SIZE + (1 - step_y) / 2) / ray_dy
            
        distance = perp_wall_dist * CELL_SIZE
        
        # Add minimum distance to prevent rendering artifacts when very close to walls
        distance = max(distance, MIN_WALL_DISTANCE)
        
        # Calculate wall height based on distance with better perspective correction
        if distance > 0:
            wall_height = int((CELL_SIZE / distance) * ((SCREEN_WIDTH / 2) / math.tan(HALF_FOV)))
        else:
            wall_height = SCREEN_HEIGHT
            
        if wall_height > SCREEN_HEIGHT:
            wall_height = SCREEN_HEIGHT
        
        # Determine wall color based on side (x or y)
        wall_color = get_wall_color(ray_angle, distance)
        if side == 1:  # Darken y-side walls to create contrast
            wall_color = tuple(int(c * 0.7) for c in wall_color)
        
        # Store wall segment data
        walls.append({
            "height": wall_height, 
            "distance": distance, 
            "color": wall_color, 
            "x": ray,
            "side": side
        })
        
        # Move to next ray
        ray_angle += FOV / SCREEN_WIDTH
    
    return walls

# Draw game world
def draw_world(walls):
    # Clear screen
    screen.fill(BLACK)
    
    # Draw sky gradient
    for y in range(SCREEN_HEIGHT // 2):
        # Create a dark blue to light blue gradient
        gradient_factor = y / (SCREEN_HEIGHT // 2)
        sky_color = (
            int(20 + 60 * gradient_factor),  # R
            int(20 + 100 * gradient_factor), # G
            int(50 + 120 * gradient_factor)  # B
        )
        pygame.draw.line(screen, sky_color, (0, y), (SCREEN_WIDTH, y))
    
    # Draw floor with simple distance shading
    for y in range(SCREEN_HEIGHT // 2, SCREEN_HEIGHT):
        # Create a distance-based floor gradient
        distance_factor = (y - SCREEN_HEIGHT // 2) / (SCREEN_HEIGHT // 2)
        distance_factor = 1 - distance_factor  # Reverse so closer is lighter
        floor_color = (
            int(40 * distance_factor),  # R
            int(40 * distance_factor),  # G
            int(40 * distance_factor)   # B
        )
        pygame.draw.line(screen, floor_color, (0, y), (SCREEN_WIDTH, y))
    
    # Draw walls
    for wall in walls:
        # Calculate wall position and size
        wall_top = (SCREEN_HEIGHT - wall["height"]) // 2
        
        # Draw wall with improved appearance (thicker lines)
        pygame.draw.line(screen, wall["color"], 
                         (wall["x"], wall_top), 
                         (wall["x"], wall_top + wall["height"]), 
                         2)  # Increased thickness for better visibility

# Draw enemies with improved visuals
def draw_enemies(walls):
    for enemy in enemies:
        if not enemy["alive"]:
            continue
            
        # Calculate enemy position relative to player
        dx = enemy["x"] - player_x
        dy = enemy["y"] - player_y
        
        # Calculate distance and angle to enemy
        distance = math.sqrt(dx**2 + dy**2)
        enemy_angle = math.atan2(dy, dx)
        
        # Adjust angle to be relative to player view
        relative_angle = enemy_angle - player_angle
        
        # Normalize angle
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        
        # Check if enemy is in view
        if abs(relative_angle) < HALF_FOV:
            # Calculate screen x-coordinate
            screen_x = int((relative_angle + HALF_FOV) / FOV * SCREEN_WIDTH)
            
            # Get enemy type properties
            enemy_type = enemy_types[enemy["type"]]
            
            # Calculate enemy size based on distance and type
            base_size = min(200, int(1500 / distance))
            enemy_size = int(base_size * enemy_type["size"])
            
            # Calculate position on screen
            enemy_top = (SCREEN_HEIGHT - enemy_size) // 2
            
            # Don't draw if behind a wall
            if distance < walls[screen_x]["distance"] * 0.8:
                # Get animation frame (0-3)
                frame = (enemy["animation_frame"] // 10) % 4
                
                # Draw enemy body (circle)
                pygame.draw.circle(screen, enemy_type["color"], (screen_x, enemy_top + enemy_size//2), enemy_size // 2)
                
                # Draw eyes based on animation frame
                eye_size = max(3, enemy_size // 8)
                eye_offset_x = enemy_size // 6
                eye_offset_y = -enemy_size // 8
                
                # Draw main eyes
                pygame.draw.circle(screen, WHITE, 
                    (screen_x - eye_offset_x, enemy_top + enemy_size//2 + eye_offset_y), 
                    eye_size)
                pygame.draw.circle(screen, WHITE, 
                    (screen_x + eye_offset_x, enemy_top + enemy_size//2 + eye_offset_y), 
                    eye_size)
                
                # Draw pupils (move based on animation frame)
                pupil_offset = frame - 1.5  # -1.5 to 1.5
                pygame.draw.circle(screen, BLACK, 
                    (int(screen_x - eye_offset_x + pupil_offset), enemy_top + enemy_size//2 + eye_offset_y), 
                    max(1, eye_size // 2))
                pygame.draw.circle(screen, BLACK, 
                    (int(screen_x + eye_offset_x + pupil_offset), enemy_top + enemy_size//2 + eye_offset_y), 
                    max(1, eye_size // 2))
                
                # Draw mouth (changes with animation)
                mouth_width = enemy_size // 3
                mouth_height = enemy_size // 8
                if frame == 0 or frame == 2:
                    # Closed angry mouth
                    pygame.draw.line(screen, BLACK,
                        (screen_x - mouth_width // 2, enemy_top + enemy_size//2 + eye_size * 2),
                        (screen_x + mouth_width // 2, enemy_top + enemy_size//2 + eye_size * 2),
                        max(1, mouth_height // 2))
                else:
                    # Open mouth
                    pygame.draw.ellipse(screen, BLACK,
                        (screen_x - mouth_width // 2, 
                         enemy_top + enemy_size//2 + eye_size * 2 - mouth_height // 2,
                         mouth_width, mouth_height))
                
                # Draw health bar above enemy
                health_width = enemy_size
                health_height = max(2, enemy_size // 10)
                health_y = enemy_top - health_height * 2
                
                # Draw health background
                pygame.draw.rect(screen, (100, 100, 100),
                    (screen_x - health_width // 2, health_y, health_width, health_height))
                
                # Draw health fill
                health_fill = (enemy["health"] / enemy_type["health"]) * health_width
                health_color = (255 - int(health_fill / health_width * 255), 
                               int(health_fill / health_width * 255), 0)
                pygame.draw.rect(screen, health_color,
                    (screen_x - health_width // 2, health_y, int(health_fill), health_height))
                
                # Increment animation frame
                enemy["animation_frame"] += 1

# Update enemy positions
def update_enemies():
    global health
    
    for enemy in enemies:
        if not enemy["alive"]:
            continue
            
        # Get enemy type properties
        enemy_type = enemy_types[enemy["type"]]
        
        # Calculate direction to player
        dx = player_x - enemy["x"]
        dy = player_y - enemy["y"]
        
        # Normalize direction
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        # Move enemy toward player at type-specific speed
        enemy["x"] += dx * enemy_type["speed"] * CELL_SIZE
        enemy["y"] += dy * enemy_type["speed"] * CELL_SIZE
        
        # Check collision with walls
        enemy_map_x = int(enemy["x"] / CELL_SIZE)
        enemy_map_y = int(enemy["y"] / CELL_SIZE)
        
        if enemy_map_x < 0 or enemy_map_x >= MAP_WIDTH or enemy_map_y < 0 or enemy_map_y >= MAP_HEIGHT or MAP[enemy_map_y][enemy_map_x] == 1:
            # Move back if colliding with wall
            enemy["x"] -= dx * enemy_type["speed"] * CELL_SIZE
            enemy["y"] -= dy * enemy_type["speed"] * CELL_SIZE
        
        # Check if enemy is too close to player
        distance = math.sqrt((player_x - enemy["x"])**2 + (player_y - enemy["y"])**2)
        if distance < CELL_SIZE * 0.5:
            # Enemy attacks player (damage based on enemy type)
            health -= enemy_type["damage"]

# Function to spawn new enemies
def spawn_enemy():
    # Find valid spawn positions (open spaces away from player)
    valid_positions = []
    min_distance = 8 * CELL_SIZE  # Minimum distance from player to spawn
    
    for y in range(1, MAP_HEIGHT - 1):
        for x in range(1, MAP_WIDTH - 1):
            # Check if position is empty
            if MAP[y][x] == 0:
                # Calculate distance to player
                spawn_x = (x + 0.5) * CELL_SIZE
                spawn_y = (y + 0.5) * CELL_SIZE
                dist_to_player = math.sqrt((spawn_x - player_x)**2 + (spawn_y - player_y)**2)
                
                # Only add if far enough from player
                if dist_to_player > min_distance:
                    valid_positions.append((spawn_x, spawn_y))
    
    # If there are valid positions, spawn an enemy
    if valid_positions:
        # Choose random position
        spawn_x, spawn_y = random.choice(valid_positions)
        
        # Choose random enemy type, weighted towards easier enemies
        weights = [0.5, 0.35, 0.15]  # 50% type 0, 35% type 1, 15% type 2
        enemy_type = random.choices([0, 1, 2], weights=weights)[0]
        
        # Create new enemy
        new_enemy = {
            "x": spawn_x,
            "y": spawn_y,
            "alive": True,
            "type": enemy_type,
            "health": enemy_types[enemy_type]["health"],
            "animation_frame": random.randint(0, 30)  # Random starting frame
        }
        
        enemies.append(new_enemy)
        return True
    
    return False

# Draw weapon with reload animation
def draw_weapon():
    global reloading
    
    # Draw a simple gun at the bottom of the screen
    gun_height = 100
    gun_width = 150
    
    # Calculate bobbing effect
    bob_offset = math.sin(pygame.time.get_ticks() * 0.005) * 5
    
    # Apply reload animation
    reload_offset = 0
    if reloading:
        # Calculate reload animation (gun moving down and up)
        reload_progress = 1 - (reload_timer / RELOAD_TIME)
        
        if reload_progress < 0.5:
            # First half - gun moves down
            reload_offset = 40 * (reload_progress * 2)
        else:
            # Second half - gun moves up
            reload_offset = 40 * (1 - (reload_progress - 0.5) * 2)
    
    # Draw gun barrel
    barrel_color = GRAY
    if reloading:
        # Flash the barrel during reload
        if (reload_timer // 5) % 2 == 0:
            barrel_color = (150, 150, 150)
    
    # Draw gun with combined effects
    total_offset = int(bob_offset + reload_offset)
    
    # Draw gun body
    pygame.draw.rect(screen, DARK_GRAY, 
                    (SCREEN_WIDTH // 2 - gun_width // 2, 
                     SCREEN_HEIGHT - gun_height + total_offset, 
                     gun_width, gun_height))
    
    # Draw gun barrel
    pygame.draw.rect(screen, barrel_color, 
                    (SCREEN_WIDTH // 2 - 10, 
                     SCREEN_HEIGHT - gun_height - 20 + total_offset, 
                     20, 30))
                     
    # Draw muzzle flash for firing
    if not reloading and ammo > 0 and pygame.mouse.get_pressed()[0] and pygame.time.get_ticks() % 10 < 5:
        # Draw a simple muzzle flash
        flash_size = random.randint(10, 20)
        pygame.draw.circle(screen, YELLOW, 
                         (SCREEN_WIDTH // 2, 
                          SCREEN_HEIGHT - gun_height - 25 + int(bob_offset)), 
                         flash_size)

# Update bullets
def update_bullets():
    global score
    
    # Move bullets
    for bullet in bullets[:]:
        bullet["x"] += bullet["dx"] * 10
        bullet["y"] += bullet["dy"] * 10
        bullet["distance"] += 10
        
        # Check if bullet hit a wall
        bullet_map_x = int(bullet["x"] / CELL_SIZE)
        bullet_map_y = int(bullet["y"] / CELL_SIZE)
        
        if (bullet_map_x < 0 or bullet_map_x >= MAP_WIDTH or 
            bullet_map_y < 0 or bullet_map_y >= MAP_HEIGHT or 
            MAP[bullet_map_y][bullet_map_x] == 1 or
            bullet["distance"] > CELL_SIZE * 5):
            bullets.remove(bullet)
            continue
        
        # Check if bullet hit an enemy
        for enemy in enemies:
            if not enemy["alive"]:
                continue
                
            # Calculate distance from bullet to enemy
            distance = math.sqrt((bullet["x"] - enemy["x"])**2 + (bullet["y"] - enemy["y"])**2)
            
            if distance < CELL_SIZE * 0.3:  # Enemy hit radius
                # Get enemy type data
                enemy_type = enemy_types[enemy["type"]]
                
                # Reduce enemy health
                enemy["health"] -= 1
                
                # Check if enemy is defeated
                if enemy["health"] <= 0:
                    enemy["alive"] = False
                    score += enemy_type["points"]
                
                # Remove bullet in any case
                if bullet in bullets:
                    bullets.remove(bullet)
                break

# Fire weapon
def fire_weapon():
    global ammo, reloading
    
    if reloading:
        return
        
    if ammo <= 0:
        # Auto reload when empty
        start_reload()
        return
        
    # Create a new bullet
    bullet = {
        "x": player_x,
        "y": player_y,
        "dx": math.cos(player_angle),
        "dy": math.sin(player_angle),
        "distance": 0
    }
    
    bullets.append(bullet)
    ammo -= 1

# Start reload animation
def start_reload():
    global reloading, reload_timer
    
    if not reloading and ammo < max_ammo:
        reloading = True
        reload_timer = RELOAD_TIME

# Update reload state
def update_reload():
    global reloading, reload_timer, ammo
    
    if reloading:
        reload_timer -= 1
        if reload_timer <= 0:
            # Reload complete
            ammo = max_ammo
            reloading = False

# Draw HUD (Heads-Up Display)
def draw_hud():
    # Draw health
    health_text = f"Health: {health}"
    health_surface = pygame.font.SysFont(None, 36).render(health_text, True, WHITE)
    screen.blit(health_surface, (20, 20))
    
    # Draw ammo with reload indicator
    if reloading:
        reload_progress = int((1 - (reload_timer / RELOAD_TIME)) * 100)
        ammo_text = f"Reloading... {reload_progress}%"
        ammo_color = YELLOW
    else:
        ammo_text = f"Ammo: {ammo}/{max_ammo}"
        ammo_color = WHITE if ammo > 0 else RED
    
    ammo_surface = pygame.font.SysFont(None, 36).render(ammo_text, True, ammo_color)
    screen.blit(ammo_surface, (20, 60))
    
    # Draw score
    score_text = f"Score: {score}"
    score_surface = pygame.font.SysFont(None, 36).render(score_text, True, WHITE)
    screen.blit(score_surface, (SCREEN_WIDTH - 150, 20))
    
    # Draw enemy count
    enemy_count = sum(1 for enemy in enemies if enemy["alive"])
    enemy_text = f"Enemies: {enemy_count}"
    enemy_surface = pygame.font.SysFont(None, 36).render(enemy_text, True, WHITE)
    screen.blit(enemy_surface, (SCREEN_WIDTH - 150, 60))
    
    # Draw crosshair
    crosshair_size = 10
    crosshair_color = RED if reloading else WHITE
    pygame.draw.line(screen, crosshair_color, 
                    (SCREEN_WIDTH // 2 - crosshair_size, SCREEN_HEIGHT // 2), 
                    (SCREEN_WIDTH // 2 + crosshair_size, SCREEN_HEIGHT // 2), 2)
    pygame.draw.line(screen, crosshair_color, 
                    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - crosshair_size), 
                    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + crosshair_size), 2)

# Game loop
def main_game_loop():
    global player_x, player_y, player_angle, health, ammo, game_over, spawn_timer, reloading, reload_timer
    
    # Add random module
    import random
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Handle shooting
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                fire_weapon()
            
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r and not game_over:
                    # Manual reload
                    start_reload()
                if event.key == pygame.K_r and game_over:
                    # Reset game
                    player_x = 3.5 * CELL_SIZE
                    player_y = 2.5 * CELL_SIZE
                    player_angle = 0
                    health = 100
                    ammo = max_ammo
                    score = 0
                    game_over = False
                    reloading = False
                    
                    # Reset enemies
                    enemies.clear()
                    for i in range(5):
                        spawn_enemy()
                        
                    # Clear bullets
                    bullets.clear()
        
        if not game_over:
            # Handle movement input
            keys = pygame.key.get_pressed()
            
            # Calculate movement vector based on keys pressed
            move_x = 0
            move_y = 0
            
            # Movement
            if keys[pygame.K_w]:  # Move forward
                move_x += math.cos(player_angle) * PLAYER_SPEED
                move_y += math.sin(player_angle) * PLAYER_SPEED
            
            if keys[pygame.K_s]:  # Move backward
                move_x -= math.cos(player_angle) * PLAYER_SPEED
                move_y -= math.sin(player_angle) * PLAYER_SPEED
            
            if keys[pygame.K_a]:  # Strafe left
                strafe_angle = player_angle - math.pi / 2
                move_x += math.cos(strafe_angle) * PLAYER_SPEED
                move_y += math.sin(strafe_angle) * PLAYER_SPEED
            
            if keys[pygame.K_d]:  # Strafe right
                strafe_angle = player_angle + math.pi / 2
                move_x += math.cos(strafe_angle) * PLAYER_SPEED
                move_y += math.sin(strafe_angle) * PLAYER_SPEED
            
            # Try to move in X direction
            new_x = player_x + move_x
            map_x = int(new_x / CELL_SIZE)
            map_y = int(player_y / CELL_SIZE)
            
            # Check player size (add buffer to prevent getting too close to walls)
            player_radius = CELL_SIZE * 0.2
            
            # Check corners around the player for X movement
            if (MAP[map_y][map_x] == 0 and
                MAP[int((player_y + player_radius) / CELL_SIZE)][map_x] == 0 and
                MAP[int((player_y - player_radius) / CELL_SIZE)][map_x] == 0):
                player_x = new_x
            
            # Try to move in Y direction
            new_y = player_y + move_y
            map_x = int(player_x / CELL_SIZE)
            map_y = int(new_y / CELL_SIZE)
            
            # Check corners around the player for Y movement
            if (MAP[map_y][map_x] == 0 and
                MAP[map_y][int((player_x + player_radius) / CELL_SIZE)] == 0 and
                MAP[map_y][int((player_x - player_radius) / CELL_SIZE)] == 0):
                player_y = new_y
            
            # Rotation with keyboard
            if keys[pygame.K_LEFT]:
                player_angle -= ROTATION_SPEED
            if keys[pygame.K_RIGHT]:
                player_angle += ROTATION_SPEED
                
            # Rotation with mouse
            mouse_rel = pygame.mouse.get_rel()
            player_angle += mouse_rel[0] * 0.003
            
            # Normalize angle
            player_angle %= 2 * math.pi
            
            # Update enemies
            update_enemies()
            
            # Update bullets
            update_bullets()
            
            # Update reload state
            update_reload()
            
            # Handle enemy spawning
            spawn_timer -= 1
            if spawn_timer <= 0:
                if spawn_enemy():
                    # Reset timer with some randomness
                    spawn_timer = SPAWN_RATE + random.randint(-SPAWN_RATE // 4, SPAWN_RATE // 4)
                else:
                    # Try again soon if no valid spawn point
                    spawn_timer = 60
            
            # Check game over condition
            if health <= 0:
                game_over = True
        
        # Draw game world
        walls = cast_rays()
        draw_world(walls)
        
        if not game_over:
            # Draw enemies
            draw_enemies(walls)
            
            # Draw weapon
            draw_weapon()
            
            # Draw HUD
            draw_hud()
            
            # Continuous fire if mouse button held down
            if pygame.mouse.get_pressed()[0] and not reloading:
                fire_weapon()
        else:
            # Display game over message
            game_over_text = "Game Over! Press R to restart"
            game_over_surface = pygame.font.SysFont(None, 72).render(game_over_text, True, RED)
            game_over_rect = game_over_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(game_over_surface, game_over_rect)
            
            # Display final score
            score_text = f"Final Score: {score}"
            score_surface = pygame.font.SysFont(None, 48).render(score_text, True, WHITE)
            score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
            screen.blit(score_surface, score_rect)
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)

# Start the game
if __name__ == "__main__":
    # Set mouse to center and hide cursor
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    # Start game loop
    main_game_loop()