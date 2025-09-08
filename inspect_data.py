import blosc2
import numpy as np
import os

# --- 設定項目 ---

# マージ済みのデータが入っているフォルダ名
DATA_DIR = "data/right"

# 中身を確認したいエピソードの番号 (0から79まで)
EPISODE_TO_CHECK = 1

# エピソードの中の、何番目のフレームを確認するか (0から49まで)
FRAME_TO_CHECK = 0 

# -----------------

def show_array_structure():
    """
    指定されたエピソードとフレームの画像データを読み込み、
    RGB各チャンネルの数値配列をターミナルに表示する。
    """
    print(f"--- Inspecting: '{DATA_DIR}', Episode {EPISODE_TO_CHECK}, Frame {FRAME_TO_CHECK} ---")
    
    try:
        images_path = os.path.join(DATA_DIR, "images_states.blosc2")
        all_images = blosc2.load_array(images_path)
        
        # 指定したエピソードとフレームが存在するかチェック
        if EPISODE_TO_CHECK >= len(all_images) or FRAME_TO_CHECK >= all_images.shape[1]:
            print("エラー: 指定したエピソードまたはフレーム番号が存在しません。")
            return
            
        # 1. 指定した1フレーム分の画像データを取得 (shape: 3, 48, 64)
        image_frame = all_images[EPISODE_TO_CHECK, FRAME_TO_CHECK]
        
        channel_names = ['Red', 'Green', 'Blue']
        
        # 2. 各チャンネルをループして表示
        for i in range(3):
            print("\n" + "="*50 + "\n")
            print(f"--- チャンネル (Channel) {i+1}: {channel_names[i]} ---")
            
            # i番目のチャンネルの配列を取得 (shape: 48, 64)
            channel_array = image_frame[i]
            print(f"Shape: {channel_array.shape}")
            
            # NumPy配列をそのままprintすると、大きい場合は自動で省略表示される
            print(channel_array)

        print("\n" + "="*50)
        print("表示完了。巨大な配列はNumPyによって自動的に '...' で省略されます。")

    except FileNotFoundError:
        print(f"エラー: データフォルダ '{DATA_DIR}' または 'images.blosc2' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    show_array_structure()