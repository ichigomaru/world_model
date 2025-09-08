import blosc2
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# --- 設定項目 ---

# ★変更点: 確認したい元のデータフォルダを指定
DATA_DIR = "data/left" 

# 中身を確認したいエピソードの番号 (0から始まる)
EPISODE_TO_CHECK = 0  # 0番目のエピソードを確認してみましょう

# ★変更点: 出力ファイル名を分かりやすく変更
OUTPUT_GIF_FILENAME = f"original_left_episode_{EPISODE_TO_CHECK}.gif"

# -----------------

def check_episode_data():
    print(f"--- 元データ '{DATA_DIR}' のエピソード {EPISODE_TO_CHECK} の検証を開始します ---")
    
    try:
        # ★変更点: 元のデータファイル名に合わせる
        images_path = os.path.join(DATA_DIR, "image_states.blosc2")
        
        print(f"'{images_path}' から画像データを読み込み中...")
        all_images = blosc2.load_array(images_path)
        
        if EPISODE_TO_CHECK >= len(all_images):
            print(f"エラー: エピソード番号 {EPISODE_TO_CHECK} は存在しません。")
            print(f"利用可能なエピソード番号は 0 から {len(all_images) - 1} までです。")
            return
            
        episode_images = all_images[EPISODE_TO_CHECK]
        print(f"エピソード {EPISODE_TO_CHECK} を取得しました。形状: {episode_images.shape}")
        
        try:
            font = ImageFont.truetype("Arial.ttf", size=15)
        except IOError:
            font = ImageFont.load_default()

        frames_for_gif = []
        for timestep, img_chw in enumerate(episode_images):
            img_hwc = np.transpose(img_chw, (1, 2, 0))
            
            # 元のデータは [0, 255] の範囲のはずなので、正規化の逆変換は不要
            # もし画像が暗い/おかしい場合は、下の行のコメントを外して正規化の逆変換を試してください
            # img_normalized = ((img_hwc + 1) / 2.0 * 255.0).clip(0, 255)
            # pil_image = Image.fromarray(img_normalized.astype(np.uint8))
            
            pil_image = Image.fromarray(img_hwc.astype(np.uint8))
            draw = ImageDraw.Draw(pil_image)
            
            text = f"Timestep: {timestep}"
            position = (5, 5)
            text_color = (255, 255, 255)
            
            draw.text(position, text, fill=text_color, font=font)
            
            frames_for_gif.append(np.array(pil_image))
            
        print(f"GIFアニメーションを '{OUTPUT_GIF_FILENAME}' として保存中...")
        imageio.mimsave(OUTPUT_GIF_FILENAME, frames_for_gif, fps=10)
        
        print("\n検証が完了しました！")
        print(f"'{OUTPUT_GIF_FILENAME}' が作成されたので、中身を確認してください。")
        
    except FileNotFoundError:
        print(f"エラー: データフォルダ '{DATA_DIR}' またはその中のファイルが見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    check_episode_data()