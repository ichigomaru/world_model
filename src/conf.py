from dataclasses import dataclass

@dataclass
class VisualizeConfig:
    act_figsize: list
    obs_figsize: list

@dataclass
class ActConfig:
    name: str
    max: list
    min: list
    use_idx_list: list

@dataclass
class MasterConfig:
    name: str
    max: list
    min: list
    use_idx_list: list

@dataclass
class VisionConfig:
    height: int
    width: int
    do_aug: bool

@dataclass
class DirConfig:
    train_dir: str
    valid_dir: str
    test_dir: str
    output_dir: str
    save_dir: str

@dataclass
class DiffuserConfig:
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    beta_schedule: str
    variance_type: str
    clip_sample: bool
    prediction_type: str

@dataclass
class PolicyConfig:
    name: str
    params: dict

@dataclass
class EncoderConfig:
    name: str
    params: dict

@dataclass
class SchedulerConfig:
    num_warmup_stps: int

@dataclass
class OptimizerConfig:
    lr: float
    betas: list
    eps: float
    weigit_decay: float

@dataclass
class TrainConfig:
    lr: float
    early_stopping_rounds: int
    epochs: int
    device: list
    accelerator: str
    debug: bool

@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    num_workers: int
    batch_size: int
    val_ratio: float
    latent_obs_dim: int
    obs_step: int
    act_step: int
    use_act_step: int
    act_dim: int
    device: str
    dir: DirConfig
    diffuser: DiffuserConfig
    encoder: EncoderConfig
    trainer: TrainConfig
    scheduler: SchedulerConfig
    optimizer: OptimizerConfig
    act: ActConfig
    master: MasterConfig
    visualize: VisualizeConfig
