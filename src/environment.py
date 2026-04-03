import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import math
import os
from gym import Env
from gym.spaces import Box


class DoublePendulumEnv(Env):
    """
    Custom Double Inverted Pendulum Environment using pymunk physics engine.
    
    The environment consists of:
    - A cart that can move horizontally on a frictionless track
    - Two poles connected in series (pole1 on cart, pole2 on pole1)
    - Goal: Balance both poles upright
    
    Observation Space: [cart_x, cart_vx, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel]
    Action Space: Continuous force in [-1.0, 1.0] applied to the cart
    """

    def __init__(self, reward_type='shaped', render_mode=None):
        """
        Initialize the Double Pendulum Environment.
        
        Args:
            reward_type (str): 'baseline' or 'shaped'. Controls reward function.
            render_mode (str): 'human' for visualization, None for headless.
        """
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        # Physics parameters
        self.dt = 1.0 / 60.0  # 60 FPS simulation
        self.gravity = 9.81
        self.max_force = 100.0  # Max force applied to cart
        self.track_length = 4.8  # Length of the track
        
        # Pole parameters
        self.pole_mass = 1.0
        self.pole_length = 0.5  # Length of each pole
        self.pole_width = 0.05
        pole_inertia = math.inf
        
        # Cart parameters
        self.cart_mass = 1.0
        self.cart_width = 0.4
        self.cart_height = 0.3
        
        # Episode parameters
        self.max_steps = 500
        self.steps = 0
        self.angle_limit_radians = 0.2 * math.pi  # Fall if > 36 degrees
        self.x_limit = self.track_length / 2.0  # Cart can't go beyond track
        
        # Pygame setup
        self.screen_width = 800
        self.screen_height = 600
        self.pixels_per_meter = 100
        self.screen = None
        self.clock = None
        self.font = None
        
        # Define action and observation spaces
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Physics space
        self.space = self._create_space()
        
        # Physical objects (will be initialized in reset)
        self.cart_body = None
        self.pole1_body = None
        self.pole2_body = None
        self.track_body = None
        
        # Reset to create initial state
        self.reset()

    def reset(self):
        """Reset the environment and return initial observation."""
        # Recreate the physics space so reset works across pymunk versions.
        self.space = self._create_space()
        
        # Create static track (ground)
        track_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        track_shape = pymunk.Segment(
            track_body, 
            (-self.track_length / 2, 0),
            (self.track_length / 2, 0),
            0.1
        )
        track_shape.friction = 0.5
        self.space.add(track_body, track_shape)
        self.track_body = track_body
        
        # Create cart
        self.cart_body = pymunk.Body(mass=self.cart_mass, moment=math.inf)
        self.cart_body.position = (0, self.cart_height / 2)
        cart_shape = pymunk.Poly.create_box(self.cart_body, (self.cart_width, self.cart_height))
        cart_shape.friction = 0.0
        self.space.add(self.cart_body, cart_shape)
        
        # Create groove joint to constrain cart to horizontal movement
        groove_joint = pymunk.GrooveJoint(
            track_body,
            self.cart_body,
            (self.cart_body.position.x, 0),
            (self.cart_body.position.x + 0.01, 0),
            (0, self.cart_height / 2)
        )
        self.space.add(groove_joint)
        
        # Create Pole 1 (attached to cart)
        self.pole1_body = pymunk.Body(mass=self.pole_mass, moment=math.inf)
        self.pole1_body.position = (0, self.cart_height + self.pole_length / 2)
        pole1_shape = pymunk.Poly.create_box(
            self.pole1_body, 
            (self.pole_width, self.pole_length)
        )
        pole1_shape.friction = 0.0
        self.space.add(self.pole1_body, pole1_shape)
        
        # Joint between cart and pole1
        pivot1 = pymunk.PivotJoint(
            self.cart_body,
            self.pole1_body,
            (0, self.cart_height)
        )
        pivot1.collide_bodies = False
        self.space.add(pivot1)
        
        # Create Pole 2 (attached to pole1)
        self.pole2_body = pymunk.Body(mass=self.pole_mass, moment=math.inf)
        self.pole2_body.position = (0, self.cart_height + self.pole_length + self.pole_length / 2)
        pole2_shape = pymunk.Poly.create_box(
            self.pole2_body, 
            (self.pole_width, self.pole_length)
        )
        pole2_shape.friction = 0.0
        self.space.add(self.pole2_body, pole2_shape)
        
        # Joint between pole1 and pole2
        pivot2 = pymunk.PivotJoint(
            self.pole1_body,
            self.pole2_body,
            (0, self.cart_height + self.pole_length)
        )
        pivot2.collide_bodies = False
        self.space.add(pivot2)
        
        self.steps = 0
        return self._get_observation()

    def _create_space(self):
        """Create a fresh physics space with the configured simulation settings."""
        space = pymunk.Space()
        space.gravity = (0, -self.gravity)
        space.damping = 0.1
        return space

    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: Continuous action in [-1.0, 1.0]
            
        Returns:
            observation, reward, done, info
        """
        # Scale action to force
        force = action[0] * self.max_force
        
        # Apply force to cart
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))
        
        # Step physics simulation
        self.space.step(self.dt)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        
        # Check if episode is done
        done = self._is_done(obs)
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return obs, reward, done, {}

    def _get_observation(self):
        """Extract state observation from physics space."""
        cart_x = self.cart_body.position.x
        cart_vx = self.cart_body.velocity.x
        
        # Get pole angles (relative to vertical, in radians)
        pole1_angle = self.pole1_body.angle
        pole1_angular_vel = self.pole1_body.angular_velocity
        
        pole2_angle = self.pole2_body.angle
        pole2_angular_vel = self.pole2_body.angular_velocity
        
        obs = np.array(
            [cart_x, cart_vx, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel],
            dtype=np.float32
        )
        return obs

    def _calculate_reward(self, obs, action):
        """
        Calculate reward based on selected reward function.
        
        Args:
            obs: Current observation
            action: Current action taken
            
        Returns:
            reward: Scalar reward value
        """
        cart_x, cart_vx, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel = obs
        force = action[0]
        
        if self.reward_type == 'baseline':
            # Baseline: Reward for keeping poles upright
            # cos(angle) gives maximum reward (1.0) when angle=0 (upright)
            upright_reward = np.cos(pole1_angle) + np.cos(pole2_angle)
            return upright_reward
        
        elif self.reward_type == 'shaped':
            # Shaped reward with multiple components
            
            # 1. Core goal: Keep poles upright
            upright_reward = np.cos(pole1_angle) + np.cos(pole2_angle)  # Max: 2.0
            
            # 2. Center penalty: Keep cart near the center of the track
            center_penalty = -np.abs(cart_x) * 0.1
            
            # 3. Velocity penalty: Penalize high angular velocities (encourages smooth motion)
            velocity_penalty = -(np.abs(pole1_angular_vel) + np.abs(pole2_angular_vel)) * 0.01
            
            # 4. Action penalty: Penalize high force usage (saves energy)
            action_penalty = -(force ** 2) * 0.001
            
            # 5. Cart velocity penalty: Penalize excessive cart movement
            cart_velocity_penalty = -np.abs(cart_vx) * 0.01
            
            total_reward = (
                upright_reward +
                center_penalty +
                velocity_penalty +
                action_penalty +
                cart_velocity_penalty
            )
            
            return total_reward
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _is_done(self, obs):
        """Check if episode should terminate."""
        cart_x, cart_vx, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel = obs
        
        # Terminate if cart goes off track
        if np.abs(cart_x) > self.x_limit:
            return True
        
        # Terminate if either pole falls over
        if np.abs(pole1_angle) > self.angle_limit_radians:
            return True
        
        if np.abs(pole2_angle) > self.angle_limit_radians:
            return True
        
        return False

    def render(self, mode='human'):
        """Render the environment using pygame."""
        if self.screen is None:
            pygame.init()
            try:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            except pygame.error:
                # Headless fallback for CI/containers with no X11 video device.
                pygame.display.quit()
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            pygame.display.set_caption("Double Pendulum Control - RL Agent")
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw track
        track_y = self.screen_height // 2
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (self.screen_width // 4, track_y),
            (3 * self.screen_width // 4, track_y),
            5
        )
        
        # Draw cart
        cart_x_screen = self.screen_width // 2 + self.cart_body.position.x * self.pixels_per_meter
        cart_y_screen = track_y
        cart_rect = pygame.Rect(
            cart_x_screen - self.cart_width * self.pixels_per_meter / 2,
            cart_y_screen - self.cart_height * self.pixels_per_meter / 2,
            self.cart_width * self.pixels_per_meter,
            self.cart_height * self.pixels_per_meter
        )
        pygame.draw.rect(self.screen, (100, 150, 255), cart_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)
        
        # Draw poles
        def draw_pole(body, color):
            center_x = self.screen_width // 2 + body.position.x * self.pixels_per_meter
            center_y = track_y - body.position.y * self.pixels_per_meter
            
            # Calculate pole endpoints
            length_pixels = self.pole_length * self.pixels_per_meter
            end_x = center_x + length_pixels * math.sin(body.angle)
            end_y = center_y - length_pixels * math.cos(body.angle)
            
            pygame.draw.line(
                self.screen,
                color,
                (center_x, center_y),
                (end_x, end_y),
                4
            )
            pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), 5)
        
        draw_pole(self.pole1_body, (255, 100, 100))
        draw_pole(self.pole2_body, (100, 100, 255))
        
        # Draw info text
        info_text = f"Steps: {self.steps} | Reward Type: {self.reward_type}"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
