import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Dueling_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 共通入力レイヤーを定義する
        self.linear1 = nn.Linear(input_size, hidden_size)        
        # Advantage Stream を定義する
        self.advantage = nn.Linear(hidden_size, output_size)
        # Value Stream を定義する
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 入力層の出力にReLU活性化関数を適用する
        x = F.relu(self.linear1(x))
        # Advantage 値を計算する
        advantage = self.advantage(x)
        # Value を計算する
        value = self.value(x)
        # ValueとAdvantageを組み合わせてQ値を求める
        return value + advantage - advantage.mean()
        
    def save(self, file_name='model.pth'):
        # モデルが存在しない場合は、ディレクトリを作成する
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # モデル状態辞書を指定したファイルに保存する
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr #学習率
        self.gamma = gamma #割引率
        self.model = model #主モデル
        # モデルに最適化器（Adam）を定義する
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 損失関数（平均二乗誤差）を定義する
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # state、next_state、action、rewardをトーチ・テンソルに変換する
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # 入力が単一のステートである場合、それを1つのバッチに整形する
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 現在の状態に対する予測 Q 値をメインモデルから取得
        pred = self.model(state)
        # 予測を複製し、ターゲットとして使用
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 次の状態について予測された最大Q値を使用して、新しいQ値を計算する
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # 選択された行動に対する目標Q値を更新する
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 偏りをゼロにし、バックプロパゲーションを行い、重みを更新する
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



