from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.lr = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.state_space = ((40*40*6*6*6*6*2))
        self.action_space = 4


        try:
            self.q_table = np.load(file_name,allow_pickle=True).item()
        except:
            self.q_table = {}


    def get_optimal_policy(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state])

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)

        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        return action

    def update_q_table(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        error = target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * error



    def getstate(self, snack, other_snake):
        state = []
        head_pos = self.head.pos
        snack_pos = snack.pos
        other_snake_head = other_snake.head.pos

        dist_from_snack = manhattan_distance(head_pos,snack_pos)
        dist_from_other_snake = manhattan_distance(head_pos,other_snake_head)

        state.append(dist_from_snack)
        state.append(dist_from_other_snake)

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for d in directions:
            next_pos = (head_pos[0]+d[0] , head_pos[1]+d[1])

            if next_pos[0] < 0 or next_pos[0] >= ROWS or next_pos[1] < 0 or next_pos[1] >= ROWS:
                state.append(0)
            elif next_pos == snack_pos:
                state.append(1)
            elif next_pos == other_snake_head:
                state.append(2)
            elif next_pos in list(map(lambda z: z.pos, self.body)):
                state.append(3)
            elif next_pos in list(map(lambda z: z.pos, other_snake.body)):
                state.append(4)
            else:
                state.append(5)

        state.append(int(len(self.body) > len(other_snake.body)))

        return tuple(state)

    def move(self, snack, other_snake):
        state = self.getstate(snack,other_snake)
        action = self.make_action(state)

        if action == 0:  # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1:  # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2:  # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3:  # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        next_state = self.getstate(snack,other_snake)
        snack, reward , win_self,win_other = self.calc_reward(snack,other_snake)
        self.update_q_table(state,action,next_state,reward)

        # if win_self or win_other:
        #     self.reset((random.randint(3,18),random.randint(3,18)))

        return state,next_state,action
        # TODO: Create new state after moving and other needed values and return them

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            reward -= 100
            win_other = True
            reset(self, other_snake)

        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 500

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 200
            win_other = True
            reset(self, other_snake)

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):

            if self.head.pos != other_snake.head.pos:
                reward -= 300
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward += 300
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    reward = 0
                    pass
                else:
                    reward -= 300
                    win_other = True

            reset(self, other_snake)

        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1


    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
