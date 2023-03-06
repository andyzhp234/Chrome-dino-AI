from mss import mss  # used to capture the screen
import pydirectinput # used to simulate key presses
import cv2 # used to process the image (frame processing)
import numpy as np # transform the image to a numpy array
import pytesseract # used to extract the text from the image
from matplotlib import pyplot as plt # visualize captured frames
import time # bring in time for pauses
from gym import Env # used to create the environment
from gym.spaces import Discrete, Box

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Building the environment
class WebGame(Env):
  def __init__(self):
    # subclass model
    super().__init__()
    self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
    self.action_space = Discrete(3)

    # define extraction parameters for the game
    self.capture = mss()
    self.game_location = {"top": 300, "left": 0, "width": 600, "height": 500}
    self.done_location = {"top": 375, "left": 630, "width": 660, "height": 70}
  
  # called to do something in the game
  def step(self, action):
    # action 0 = Jump (space) action 1 = Duck (down) action 2 = No action
    action_map = {
      0: 'space',
      1: 'down',
      2: 'no_op'
    }

    if action != 2:
      pydirectinput.press(action_map[action])

    # checking whether the game is done
    done, done_cap = self.get_done()
    new_obs = self.get_observation()
    reward = 1
    info = {}
    return new_obs, reward, done, info

  # Visualize the game
  def render(self):
    cv2.imshow('game', np.array(self.capture.grab(self.game_location))[:,:,:3])
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      self.close()
      
  # close observation
  def close(self):
    cv2.destroyAllWindows()
      
  # Reset the game
  def reset(self):
    time.sleep(1)
    pydirectinput.click(x=150, y=150)
    pydirectinput.press('space')
    return self.get_observation()

  # Get the part of the game that we want to use
  def get_observation(self):
    # get the screen capture of game
    raw = np.array(self.capture.grab(self.game_location))[:,:,:3]
    # Grayscale 
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    # Resize
    resized = cv2.resize(gray, (100, 83))
    # Add channel first 
    channel = np.reshape(resized, (1, 83, 100))
    return channel

  # Get the done text using OCR
  def get_done(self):
      done_cap = np.array(self.capture.grab(self.done_location))
      done_strings = ['GAME', 'GAHE', 'GANE']
      done=False
      res = pytesseract.image_to_string(done_cap)[:4]
      print(res)
      if res in done_strings:
          done = True
      return done, done_cap


env = WebGame()

import os
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common import env_checker

# env_checker.check_env(env)
class TrainAndLoggingCallback(BaseCallback):
  def __init__(self, check_freq, save_path, verbose=1):
    super(TrainAndLoggingCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.save_path = save_path

  def _init_callback(self):
    if self.save_path is not None:
      os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self):
    if self.n_calls % self.check_freq == 0:
      model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
      self.model.save(model_path)
    return True

CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# Import DQN Algorithm
from stable_baselines3 import DQN
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, buffer_size=1200000, learning_starts=1000)

# Kick off training
model.learn(total_timesteps=5000, callback=callback)
