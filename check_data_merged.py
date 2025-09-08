import blosc2
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# --- 設定項目 ---
DATA_DIR = "data_merged1"
EPISODE_TO_CHECK = 1
CANVAS_WIDTH = 128
CANVAS_HEIGHT = 96
MARGIN_COLOR = "white"
OUTPUT_GIF_FILENAME = f"merged_episode_{EPISODE_TO_CHECK}_with_margin.gif"
# -----------------

def check_episode_data():
    print(f"--- マージ後データ '{DATA_DIR}' のエピソード {EPISODE_TO_CHECK} の検証を開始します ---")
    
    try:
        images_path = os.path.join(DATA_DIR, "images.blosc2")
        print(f"'{images_path}' から画像データを読み込み中...")
        all_images = blosc2.load_array(images_path)
        
        if EPISODE_TO_CHECK >= len(all_images):
            print(f"エラー: エピソード番号 {EPISODE_TO_CHECK} は存在しません。")
            return
            
        episode_images = all_images[EPISODE_TO_CHECK]
        print(f"エピソード {EPISODE_TO_CHECK} を取得しました。形状: {episode_images.shape}")
        
        try:
            font = ImageFont.truetype("Arial.ttf", size=12)
        except IOError:
            font = ImageFont.load_default()

        frames_for_gif = []
        for timestep, img_chw in enumerate(episode_images):
            img_hwc = np.transpose(img_chw, (1, 2, 0))

            # 不要な計算を削除し、直接画像に変換する
            pil_image = Image.fromarray(img_hwc.astype(np.uint8))

            draw = ImageDraw.Draw(pil_image)
            
            canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), MARGIN_COLOR)
            
            img_w, img_h = pil_image.size
            position_x = (CANVAS_WIDTH - img_w) // 2
            position_y = (CANVAS_HEIGHT - img_h) // 2
            canvas.paste(pil_image, (position_x, position_y))

            draw = ImageDraw.Draw(canvas)
            text = f"Timestep: {timestep}"
            text_width = draw.textlength(text, font=font)
            text_position_x = (CANVAS_WIDTH - text_width) // 2
            text_position_y = position_y + img_h + 5
            
            # ★★★ここが修正点★★★
            # 座標を (x, y) のタプルとして渡す
            draw.text((text_position_x, text_position_y), text, fill="black", font=font)
            
            frames_for_gif.append(np.array(canvas))
            
        print(f"GIFアニメーションを '{OUTPUT_GIF_FILENAME}' として保存中...")
        imageio.mimsave(OUTPUT_GIF_FILENAME, frames_for_gif, fps=10)
        
        print("\n検証が完了しました！")
        print(f"'{OUTPUT_GIF_FILENAME}' が作成されたので、中身を確認してください。")
        
    except FileNotFoundError:
        print(f"エラー: データフォルダ '{DATA_DIR}' または 'images.blosc2' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    check_episode_data()