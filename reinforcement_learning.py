import numpy as np

# Định nghĩa môi trường Grid World
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # Các ô chướng ngại vật

        for obstacle in self.obstacles:
            self.grid[obstacle[0], obstacle[1]] = -1

    def is_valid_move(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != -1

    def get_reward(self, position):
        if position == self.goal:
            return 10
        elif position in self.obstacles:
            return -5
        else:
            return 0

# Thuật toán Q-learning
def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, num_episodes=1000):
    q_table = np.zeros((env.size, env.size, 4))  # 4 actions: UP, DOWN, LEFT, RIGHT

    for _ in range(num_episodes):
        state = env.start
        while state != env.goal:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4)  # Chọn action ngẫu nhiên
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state = (state[0] - 1, state[1]) if action == 0 else \
                         (state[0] + 1, state[1]) if action == 1 else \
                         (state[0], state[1] - 1) if action == 2 else \
                         (state[0], state[1] + 1)  # UP, DOWN, LEFT, RIGHT

            if not env.is_valid_move(next_state):
                continue

            reward = env.get_reward(next_state)
            future_reward = np.max(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] += learning_rate * (reward + discount_factor * future_reward - q_table[state[0], state[1], action])
            state = next_state

    return q_table

# Thử nghiệm chương trình
if __name__ == "__main__":
    env = GridWorld(size=5)
    q_table = q_learning(env)

    print("Q-Table:")
    print(q_table)
