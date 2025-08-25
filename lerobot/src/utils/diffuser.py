from diffusers import DDiMScheduler, DDPMScheduler


def diffuser_mode(name):
    if name == "DDIM":
        diffuser = DDiMScheduler(
            num_train_timesteps=100,  # Diffusionのステップ数
            beta_start=0.0001,  # Diffusionのbetaの開始値
            beta_end=0.02,  # Diffusionのbetaの終了値
            beta_schedule="linear",  # Diffusionのbetaのスケジュール
            clip_sample=True,  # Diffusionのサンプルをクリップするか
            prediction_type="epsilon",  # Diffusionの予測タイプ
        )
    else:
        diffuser = DDPMScheduler(
            num_train_timesteps=100,  # Diffusionのステップ数
            beta_start=0.0001,  # Diffusionのbetaの開始値
            beta_end=0.02,  # Diffusionのbetaの終了値
            beta_schedule="linear",  # Diffusionのbetaのスケジュール
            clip_sample=True,  # Diffusionのサンプルをクリップするか
            prediction_type="epsilon",  # Diffusionの予測タイプ
        )
    return diffuser
