import torch
from torch import nn


class FeedforwardModel(nn.Module):

    def __init__(self, n_stores, n_weekday, n_monthday,
                 input_dim, hidden_size,
                 n_store_features=10,
                 n_weekday_features=2,
                 n_monthday_features=5):
        super(FeedforwardModel, self).__init__()
        self.store_embedding = nn.Embedding(n_stores, n_store_features)
        self.weekday_embedding = nn.Embedding(n_weekday, n_weekday_features)
        self.month_embedding = nn.Embedding(n_monthday, n_monthday_features)
        n_inputs = (n_store_features + n_weekday_features + n_monthday_features
                    + input_dim - 3)

        self.predictor = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.n_stores = n_stores
        self.n_weekday = n_weekday
        self.n_monthday = n_monthday
        self.input_dim = input_dim
        self.n_store_features = n_store_features
        self.hidden_size = hidden_size

    def forward(self, inputs):
        store_idx = inputs[:, 0].type(torch.long)
        week_idx = inputs[:, -2].type(torch.long)
        month_idx = inputs[:, -1].type(torch.long)

        embedded_store = self.store_embedding(store_idx)
        embedded_weekday = self.weekday_embedding(week_idx)
        embedded_monthday = self.month_embedding(month_idx)

        all_features = torch.cat(
            (embedded_store, embedded_weekday, embedded_monthday,
             inputs[:, 1:-2]), dim=1)
        # print('all_features.size() =', all_features.size())
        # print('expecte_inp', self.predictor[0].weight.size())
        return self.predictor(all_features)
