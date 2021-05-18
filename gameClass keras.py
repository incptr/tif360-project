import numpy as np
import random
import copy
import keras
import tensorflow as tf
import connect4models
import connect4encoders
import Connect4
import mcts
import mcstagent


class randomagent:
    def __init__(self):
        self.num_of_games = 0
        self.train_mode = False #Dummy variable
    
    def choose_move(self, game):               
        action = random.choice(game.get_valid_moves())
        return [], int(action)
    
    def train(self, batch):
        pass

class Connect4Agent:
    def __init__(self):
        #self.game = game
        self.epsilon = 0.1
        self.Q_nn = connect4models.connectzero()
        self.epsilon_scale = 1000
        self.num_of_games = 0      
        self.encoder = connect4encoders.encoder()
        self.train_mode = True


    def choose_move(self, g):               
        root = mcts.mcts_node(g, self.Q_nn)
        policy = mcts.monte_carlo_tree_search(root)
        if random.uniform(0, 1) < self.epsilon and self.train_mode: #max(self.epsilon, 1-self.num_of_games/self.epsilon_scale) and self.train_mode: 
            move = random.choice(g.get_valid_moves())
        else :
            valid_moves = g.get_valid_moves()
            moveid = np.argmax(np.array(policy)[valid_moves])
            move = valid_moves[moveid]
        return policy, int(move)

    def train(self, batch):
        x_train = []
        policy_train = []
        value_train = []
        print(len(batch))
        for (game, policy, reward) in batch :
            x_train.append(self.encoder.encode_state(game.board, game.player))
            y_tmp = policy
            policy_train.append(y_tmp)
            value_train.append(reward)
        self.Q_nn.fit(np.array(x_train), [np.array(policy_train), np.array(value_train)], batch_size = 32, shuffle= True, epochs = 1, verbose = 0)


class experience:
    def __init__(self):
        self.exp = []

    def add_new_experience(self, new_data):
        self.exp.extend(new_data)
        if len(self.exp) > 10000:
            self.exp = self.exp[len(self.exp)-10000:]

    def get_training_batch(self, batchsize = 1024):
        return random.sample(self.exp, min(batchsize, len(self.exp))) 


exp_buffer = experience()
game = Connect4.Game(random.choice([1,2]))
best_agent = Connect4Agent()
current_agent = Connect4Agent()


def eval(num_of_games):
    best_agent_wins = 0
    current_agent_wins = 0
    num_ties = 0
    best_agent.train_mode = False
    current_agent.train_mode = False
    for _ in range(num_of_games):
        game.new_game(random.choice([1,2]))
        while True:
            if game.player == 1 :
                _, move = best_agent.choose_move(game)
                game.make_move(move)
            elif game.player == 2 :
                _, move = current_agent.choose_move(game)
                game.make_move(move)       

            if game.terminal_state:
                if game.winner == 1:
                    best_agent_wins +=1
                elif game.winner == 2: 
                    current_agent_wins +=1
                else:
                    num_ties += 1
                break
        print('')
        game.print_board()
        print('')
    best_agent.train_mode = True
    current_agent.train_mode = True
    return best_agent_wins, current_agent_wins, num_ties 




def create_training_data(num_of_games):
    for _ in range (num_of_games):
        game.new_game(random.choice([1,2]))
        episode_buffer1 = []
        episode_buffer2 = []
        while True: 
            if game.player == 1 :
                policy, move = best_agent.choose_move(game)
                episode_buffer1.append([copy.deepcopy(game), policy, 0])
                game.make_move(move)
            elif game.player == 2 :
                policy, move = best_agent.choose_move(game)
                episode_buffer2.append([copy.deepcopy(game), policy, 0])
                game.make_move(move)

            if game.terminal_state:
                if game.winner == 1:
                    for i in episode_buffer1 :
                        i[2] = 1
                    for i in episode_buffer2 :
                        i[2] = -1
                    break
                elif game.winner == 2: 
                    for i in episode_buffer1 :
                        i[2] = -1
                    for i in episode_buffer2 :
                        i[2] = 1
                    break

        exp_buffer.add_new_experience(episode_buffer1)
        exp_buffer.add_new_experience(episode_buffer2)
        best_agent.num_of_games +=1
        current_agent.num_of_games +=1
        episode_buffer1 = []
        episode_buffer2 = []
    
    return exp_buffer.get_training_batch()



# Main loop
while True:
    for i in range (10):
        print('Creating training_data')
        training_data = create_training_data(25)
        print('Training Current_agent')
        current_agent.train(training_data)
        print('##### evaluation #######')
        best_agent_wins, current_agent_wins, num_ties = eval(10)
        print(f'The best agent won {best_agent_wins} times')
        print(f'The current agent won {current_agent_wins} times')
        print(f'The game ended in a tie {num_ties} times')
        win_ratio = current_agent_wins/10
        if win_ratio > 0.55:
            print('copy nn')
            best_agent.Q_nn.set_weights(current_agent.Q_nn.get_weights())
    best_agent.Q_nn.save("best_model")
    current_agent.Q_nn.save("best_model")
    c = input('Continue?')
    if c == 'n':
        break

s = input('Save the models?: ')
if s == 'y':  
    best_agent.Q_nn.save("best_model")
    current_agent.Q_nn.save("best_model")


best_agent.Q_nn = keras.models.load_model("best_model")

# Play against the Policy network

game.new_game(random.choice([1,2]))
game.print_board()
agent1.train_mode = False

while True:
    while True:
        if game.player == 1 :
            _, move = best_agent.choose_move(game)
            game.make_move(move)
            print('')
            game.print_board()
            print('')
            game.player = 2
        elif game.player == 2 :
            move = int(input('Make your move, (0-6): '))
            game.make_move(move)
            print('')
            game.print_board()
            print('')
            game.player = 1 


        if game.terminal_state:
            if game.winner == 1:
                print('Pleyer 1 won (bot).')
            elif game.winner == 2: 
                print('Pleyer 2 won (you).')
            else:
                print('It was a tie.')
            break



    again = input('again? (y or n)')
    if again == 'y':
        game.new_game(random.choice([1,2]))
        game.print_board()
    else:
        break

