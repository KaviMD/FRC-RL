import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

SCALE = 30
BOX_SIZE = 1
VIEWPORT_W = 600
VIEWPORT_H = 600
FPS = 50
INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder
SPEED_MULT = 10
TURN_MULT = 5
MAX_TIME = 10 * FPS


class DefensePractice(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World((0,0))
        self.red_bot = None
        self.blue_bot = None

        self.time_penalty = 0
        self.prev_red_reward = 0

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

        #self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _destroy(self):
        if not self.red_bot: return
        self.world.DestroyBody(self.red_bot)
        self.red_bot = None
        self.world.DestroyBody(self.blue_bot)
        self.blue_bot = None

    def reset(self):
        self._destroy()
        self.game_over = False
        self.time_penalty = 0
        self.prev_red_reward = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE
        
        self.red_bot = self.world.CreateDynamicBody(
            position=(W/2, H/2-BOX_SIZE),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(box=(BOX_SIZE,BOX_SIZE)),
                density=10.0))
        self.red_bot.color1 = (0.9, 0.4, 0.5)
        self.red_bot.color2 = (0.5, 0.3, 0.3)
        
        self.blue_bot = self.world.CreateDynamicBody(
            position=(W/2, H/2+BOX_SIZE),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(box=(BOX_SIZE,BOX_SIZE)),
                density=10.0))
        self.blue_bot.color1 = (0.5, 0.4, 0.9)
        self.blue_bot.color2 = (0.3, 0.3, 0.5)

        self.drawlist = [self.red_bot, self.blue_bot]

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):
        action = np.clip(action, -1, +1)

        # Move object at set velocity: https://www.iforce2d.net/b2dtut/constant-speed

        # Update Red Robot
        angularVelocityAdjustment = (action[1]*TURN_MULT-self.red_bot.angularVelocity)*self.red_bot.mass
        self.red_bot.ApplyAngularImpulse(angularVelocityAdjustment, True)

        baseVelocity = np.array([math.cos(self.red_bot.angle),math.sin(self.red_bot.angle)])*action[0]*SPEED_MULT

        linearVelocityAdjustment = (baseVelocity-self.red_bot.linearVelocity)*self.red_bot.mass
        self.red_bot.ApplyLinearImpulse(linearVelocityAdjustment, (self.red_bot.position[0], self.red_bot.position[1]), True)

        # Update Blue Robot
        # angularVelocityAdjustment = (action[3]*TURN_MULT-self.blue_bot.angularVelocity)*self.blue_bot.mass
        # self.blue_bot.ApplyAngularImpulse(angularVelocityAdjustment, True)

        # baseVelocity = np.array([math.cos(self.blue_bot.angle),math.sin(self.blue_bot.angle)])*action[2]*SPEED_MULT

        # linearVelocityAdjustment = (baseVelocity-self.blue_bot.linearVelocity)*self.blue_bot.mass
        # self.blue_bot.ApplyLinearImpulse(linearVelocityAdjustment, (self.blue_bot.position[0], self.blue_bot.position[1]), True)

        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        self.time_penalty += 0.01

        done = False
        if not 0 < self.red_bot.position[0] < VIEWPORT_W/SCALE:
            done = True
        if not 0 < self.red_bot.position[1] < VIEWPORT_H/SCALE:
            done = True
        if not 0 < self.blue_bot.position[0] < VIEWPORT_W/SCALE:
            done = True
        if not 0 < self.blue_bot.position[1] < VIEWPORT_H/SCALE:
            done = True
        
        red_pos_normalized = self.red_bot.position/(VIEWPORT_W/SCALE)
        red_linVel_normalized = self.red_bot.linearVelocity/SPEED_MULT
        red_state = np.array([
            red_pos_normalized[0]-0.5,
            red_pos_normalized[1]-0.5,
            self.red_bot.angle,
            red_linVel_normalized[0],
            red_linVel_normalized[1]
        ])

        blue_pos_normalized = self.blue_bot.position/(VIEWPORT_W/SCALE)
        blue_linVel_normalized = self.blue_bot.linearVelocity/SPEED_MULT
        blue_state = np.array([
            blue_pos_normalized[0]-0.5,
            blue_pos_normalized[1]-0.5,
            self.blue_bot.angle,
            blue_linVel_normalized[0],
            blue_linVel_normalized[1]
        ])

        robot_dist = (np.linalg.norm(self.red_bot.position-self.blue_bot.position))/(VIEWPORT_W/SCALE)
        blue_pos = (self.blue_bot.position[0]-(VIEWPORT_W/SCALE)/2)/((VIEWPORT_W/SCALE)/2)

        #distance_penalty = 0.5
        #red_reward = blue_pos * (1-robot_dist)

        #if blue_pos > (VIEWPORT_W/SCALE)/2:
        #    red_reward += 100
        
        #turn_penalty = 0.5
        #red_reward = red_reward - self.time_penalty - (abs(action[1]))*turn_penalty
        #red_reward = 1-robot_dist

        red_reward = self.red_bot.position[0]#/(VIEWPORT_W/SCALE)

        #return np.concatenate([red_state,blue_state]), red_reward, np.concatenate([blue_state,red_state]), blue_reward, done, {}
        #return np.concatenate([red_state,blue_state]), red_reward, done, {}
        return red_state, red_reward, done, {}
    
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    env = DefensePractice()
    env.reset()
    for _ in range(MAX_TIME):
        env.render()
        done = env.step([0,-1,0,0])[2]
        #done = env.step(env.action_space.sample())[2] # take a random action
        if done:
            break
    env.close()
