import src.Network.GeneralNetwork as gNet
import torch.nn as nn


def init_weights_lstm(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
        # m.weight.data.fill_(0.5)
        m.bias.data.fill_(0.1)
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'bias' in name:
                pass
                nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                # nn.init.constant_(param, 0.5)
                nn.init.xavier_uniform_(param)


def init_weights_fc(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        # m.weight.data.fill_(0.5)
        m.bias.data.fill_(0.1)


class RegressionNet(gNet.GeneralNetwork):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        # model
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=self.day_size, hidden_size=self.hparams["n_hidden"], num_layers=self.hparams["layers"]),
        )

        self.fc1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            nn.BatchNorm1d(self.hparams["n_hidden"]),
            nn.PReLU(),
            # nn.Dropout(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hparams["n_hidden"], self.num_classes)
        )

        # initialize weights
        # MUSS ÜBERPRFÜFT WERDEN
        self.lstm.apply(init_weights_lstm)
        self.fc1.apply(init_weights_lstm)
        self.fc2.apply(init_weights_fc)

    def forward(self, x, *args, **kwargs):
        out, _ = self.lstm(x.permute(1, 0, 2))

        x = self.fc1(out[-1])
        x = self.fc2(x)

        return x

    def compute_acc(self, out, target):
        acc = (1 - abs(target - out)).mean()
        return acc
