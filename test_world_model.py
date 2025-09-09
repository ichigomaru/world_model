import os
import yaml
import blosc2
from box import Box
import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.model.world_model.encoder import VisionEncoder
from src.model.world_model.RSSM import RSSM
from src.model.world_model.decoder import VisionDecoder
from src.model.world_model.WorldModel import WorldModel

with open("conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

def tensor_to_pil(tensor):
    img_np = tensor.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    # [-1, 1] から [0, 255] にスケールを戻す
    img_np = ((img_np + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def create_comparison_frame(original_img, predicted_img, timestep):
    # ★変更点: 画像を何倍に拡大するか設定 (例: 4倍)
    SCALE_FACTOR = 4

    # ★変更点: 元の画像を拡大する
    # Image.Resampling.NEAREST は、ドット絵のようにくっきりと拡大する指定
    large_original_img = original_img.resize(
        (original_img.width * SCALE_FACTOR, original_img.height * SCALE_FACTOR),
        Image.Resampling.NEAREST
    )
    large_predicted_img = predicted_img.resize(
        (predicted_img.width * SCALE_FACTOR, predicted_img.height * SCALE_FACTOR),
        Image.Resampling.NEAREST
    )

    # ★変更点: 拡大後の画像サイズを基準にキャンバスを作成
    img_width, img_height = large_original_img.size
    
    # --- レイアウト設定 ---
    margin = 20
    top_margin = 60
    bottom_margin = 40
    canvas_width = img_width * 2 + margin * 3
    canvas_height = img_height + top_margin + bottom_margin
    
    try:
        font = ImageFont.truetype("Arial.ttf", size=24)
        label_font = ImageFont.truetype("Arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # ★変更点: 拡大後の画像をキャンバスに配置
    canvas.paste(large_original_img, (margin, top_margin))
    canvas.paste(large_predicted_img, (img_width + margin * 2, top_margin))

    timestep_text = f"Timestep: {timestep}"
    # Pillow 9.2.0以降では textlength は非推奨のため getlength を使用
    try:
        text_width = draw.textlength(timestep_text, font=font)
    except AttributeError:
        text_width = draw.getlength(timestep_text, font=font)
    draw.text(((canvas_width - text_width) / 2, 15), timestep_text, fill="black", font=font)

    original_text = "Original"
    try:
        text_width = draw.textlength(original_text, font=label_font)
    except AttributeError:
        text_width = draw.getlength(original_text, font=label_font)
    draw.text((margin + (img_width - text_width) / 2, top_margin + img_height + 5), original_text, fill="black", font=label_font)

    prediction_text = "Prediction"
    try:
        text_width = draw.textlength(prediction_text, font=label_font)
    except AttributeError:
        text_width = draw.getlength(prediction_text, font=label_font)
    draw.text((img_width + margin * 2 + (img_width - text_width) / 2, top_margin + img_height + 5), prediction_text, fill="black", font=label_font)

    return canvas

def imagine_future(model, image_sequence, action_sequence):
    model.eval()
    device = next(model.parameters()).device
    
    comparison_frames = []
    
    with torch.no_grad():
        # 1. 最初の1フレームから、未来予測の起点となる状態(h, z)を取得する
        first_frame = image_sequence[0]
        initial_obs_seq = first_frame.unsqueeze(0).unsqueeze(0).to(device)
        action_size = action_sequence.shape[-1]
        dummy_action_seq = torch.zeros(1, 1, action_size).to(device)
        
        _, deterministic_states, stochastic_states, _, _ = model(initial_obs_seq, dummy_action_seq)
        
        initial_deterministic = deterministic_states.squeeze(1)
        initial_stochastic = stochastic_states.squeeze(1)

        # 2. 未来の行動シーケンスを準備する
        print("Dreaming future states...")
        action_sequence_dev = action_sequence.unsqueeze(0).to(device)

        # 3. dreamメソッドを「1回だけ」呼び出し、未来の状態(h, z)を想像させる
        imagined_det_states, imagined_stoch_states = model.rssm.dream(
            initial_deterministic, 
            initial_stochastic, 
            action_sequence_dev
        )
        
        # 4. 想像した未来の状態(h+z)から、未来の画像を復元する
        print("Decoding and creating comparison frames...")
        latent_states_for_decoder = torch.cat([imagined_det_states, imagined_stoch_states], dim=-1)
        b, t, s = latent_states_for_decoder.shape
        h, w = cfg.model.vision.input_size 
        imagined_images_tensor = model.vision_decoder(latent_states_for_decoder.reshape(b * t, s))
        imagined_images_tensor = imagined_images_tensor.reshape(b, t, 3, h, w).squeeze(0)

        # 5. GIFを作成する (この部分は変更なし)
        # 最初のフレーム
        original_pil = tensor_to_pil(image_sequence[0])
        comparison_frames.append(create_comparison_frame(original_pil, original_pil, 0))

        # 2フレーム目以降
        for i in tqdm(range(t - 1)):
            original_pil = tensor_to_pil(image_sequence[i + 1])
            predicted_pil = tensor_to_pil(imagined_images_tensor[i])
            frame = create_comparison_frame(original_pil, predicted_pil, i + 1)
            comparison_frames.append(frame)

    return comparison_frames

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with open("conf/conf.yaml", "r") as yml:
        cfg = Box(yaml.safe_load(yml))

    encoder = VisionEncoder(
        channels=cfg.model.vision.channels, 
        kernels=cfg.model.vision.kernels,
        strides=cfg.model.vision.strides, 
        paddings=cfg.model.vision.paddings,
        latent_obs_dim=cfg.model.latent_obs_dim,
        mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
        n_mlp_layers=cfg.model.mlp.n_mlp_layers
    )
    rssm = RSSM(action_size=6, config=cfg)
    decoder = VisionDecoder(
        channels=cfg.model.vision.channels, 
        kernels=cfg.model.vision.kernels,
        strides=cfg.model.vision.strides, 
        paddings=cfg.model.vision.paddings,
        latent_obs_dim=cfg.parameters.dreamer.deterministic_size + cfg.parameters.dreamer.stochastic_size, 
        mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
        n_mlp_layers=cfg.model.mlp.n_mlp_layers
    )
    world_model = WorldModel(encoder, rssm, decoder).to(device)

    model_path = f"result/{cfg.wandb.train_name}/model/world_model_best.safetensors"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}")
    state_dict = load_file(model_path, device="cpu")
    world_model.load_state_dict(state_dict)
    world_model.to(device)

    data_dir = "data_merged1" # ★あなたのマージ済みデータフォルダを指定
    print(f"Loading preprocessed data from {data_dir}")
    images_np = blosc2.load_array(os.path.join(data_dir, "images.blosc2"))
    actions_np = blosc2.load_array(os.path.join(data_dir, "actions.blosc2"))

    images = torch.from_numpy(images_np).float()
    actions = torch.from_numpy(actions_np).float()
    
    images = (images / 127.5) - 1.0

    episode_index = 1

    image_sequence = images[episode_index]
    action_sequence = actions[episode_index]

    comparison_frames = imagine_future(world_model, image_sequence, action_sequence)

    output_dir = f"output/{cfg.wandb.train_name}_imagined"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"imagined_future_start_{episode_index}.gif")
    print(f"Saving comparison GIF to {gif_path}")
    
    comparison_frames[0].save(
        gif_path,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=200,  # 5fps
        loop=0
    )

if __name__ == "__main__":
    main()
