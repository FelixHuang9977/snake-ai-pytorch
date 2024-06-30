import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 最初の線形レイヤーを定義する
        self.linear1 = nn.Linear(input_size, hidden_size)
        # 2番目の線形層を定義する
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 第1層の出力にReLU活性化関数を適用する
        x = F.relu(self.linear1(x))
        # 活性化関数を使用せずに第2層を処理
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # モデルを保存するディレクトリが存在しない場合は作成する
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # モデル状態辞書を指定されたファイルに保存する
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # 学習率
        self.gamma = gamma # 割引率
        self.model = model
        # モデルに最適化器（Adam）を定義する
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 損失関数（平均二乗誤差）を定義する
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # state、next_state、action、reward を torch テンソルに変換する
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

        # 1: 現在の状態における予測Q値
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(次の予測Q値) -> まだ実行されていない場合のみ行う
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



