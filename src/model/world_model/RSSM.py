import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class RSSM(nn.Module):

    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer
        self.rssm_config = self.config.rssm

        # 1. 決定的状態を更新するRNN (GRU)
        self.recurrent_model = nn.GRUCell(
            input_size=self.config.stochastic_size + action_size,
            hidden_size=self.config.deterministic_size
        )

        # 2. 事前分布 (Prior) を予測
        self.transition_model = build_network(
            input_size=self.config.deterministic_size,
            hidden_size=self.rssm_config.transition_model.hidden_size,
            num_layers=self.rssm_config.transition_model.num_layers,
            activation=self.rssm_config.transition_model.activation,
            output_size=self.config.stochastic_size * 2 # meanとstd
        )

        # 3. 事後分布 (Posterior) を計算
        self.representation_model = build_network(
            input_size=self.config.deterministic_size + self.config.embedded_state_size,
            hidden_size=self.rssm_config.representation_model.hidden_size,
            num_layers=self.rssm_config.representation_model.num_layers,
            activation=self.rssm_config.representation_model.activation,
            output_size=self.config.stochastic_size * 2 # meanとstd
        )

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.config.deterministic_size, device=device),
            torch.zeros(batch_size, self.config.stochastic_size, device=device)
        )

    def forward(self, embedded_obs, actions):

        batch_size, seq_len, _ = embedded_obs.shape
        
        # 内部でループを回してシーケンスを処理
        deterministic_state, stochastic_state = self.initial_state(batch_size)
        
        det_states, stoch_states, priors, posteriors = [], [], [], []

        for t in range(seq_len):
            # 1. 決定的状態を更新
            deterministic_state = self.recurrent_model(
                torch.cat([stochastic_state, actions[:, t]], dim=-1),
                deterministic_state
            )
            
            # 2. 事前分布 (Prior) を予測
            prior_params = self.transition_model(deterministic_state)
            prior_dist = create_normal_dist(prior_params, min_std=self.rssm_config.transition_model.min_std)
            
            # 3. 事後分布 (Posterior) を計算
            posterior_params = self.representation_model(
                torch.cat([deterministic_state, embedded_obs[:, t]], dim=-1)
            )
            posterior_dist = create_normal_dist(posterior_params, min_std=self.rssm_config.representation_model.min_std)
            
            # 4. 事後分布から次の確率的状態をサンプリング
            stochastic_state = posterior_dist.rsample()
            
            # 結果を保存
            det_states.append(deterministic_state)
            stoch_states.append(stochastic_state)
            priors.append(prior_dist)       # ここではまだリストに分布オブジェクトを追加
            posteriors.append(posterior_dist) # ここではまだリストに分布オブジェクトを追加

        # --- ここからが変更点 ---
        # ループが終わった後、リスト内の分布からmeanとscaleをそれぞれ取り出してスタックする
        
        # Priorsを1つの大きな分布にまとめる
        prior_means = torch.stack([dist.mean for dist in priors], dim=1)
        prior_stds = torch.stack([dist.scale for dist in priors], dim=1)
        priors_dist = Normal(prior_means, prior_stds)

        # Posteriorsを1つの大きな分布にまとめる
        posterior_means = torch.stack([dist.mean for dist in posteriors], dim=1)
        posterior_stds = torch.stack([dist.scale for dist in posteriors], dim=1)
        posteriors_dist = Normal(posterior_means, posterior_stds)

        # 結果をテンソルと、1つの大きな分布オブジェクトとして返す
        return (
            torch.stack(det_states, dim=1),
            torch.stack(stoch_states, dim=1),
            priors_dist,     # 変更点：リストではなく、1つの分布オブジェクトを返す
            posteriors_dist  # 変更点：リストではなく、1つの分布オブジェクトを返す
        )

    def dream(self, initial_deterministic, initial_stochastic, action_sequence):

        deterministic_state = initial_deterministic
        stochastic_state = initial_stochastic
        
        imagined_stoch_states = []
        
        for t in range(action_sequence.size(1)):
            # 1. 決定的状態を更新
            deterministic_state = self.recurrent_model(
                torch.cat([stochastic_state, action_sequence[:, t]], dim=-1),
                deterministic_state
            )
            
            # 2. 事前分布 (Prior) から次の状態を想像
            prior_params = self.transition_model(deterministic_state)
            prior_dist = create_normal_dist(prior_params, min_std=self.rssm_config.transition_model.min_std)
            stochastic_state = prior_dist.rsample()
            
            imagined_stoch_states.append(stochastic_state)
            
        return torch.stack(imagined_stoch_states, dim=1)

#ヘルパー関数
def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist

