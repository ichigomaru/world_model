import blosc2
import numpy as np
import os

# --- 設定項目 ---
# 左腕のデータが入っている「フォルダ名」を指定
LEFT_DATA_DIR = "data/left"
# 右腕のデータが入っている「フォルダ名」を指定
RIGHT_DATA_DIR = "data/right"

# 出力される新しい「フォルダ名」を指定
OUTPUT_DIR = "data_merged2"

# それぞれのデータから取り出すエピソード（まとまり）の数
NUM_EPISODES_EACH = 40
# -----------------

def merge_datasets():
    print("データセットのマージを開始します...")

    try:
        # --- データを読み込む ---
        print(f"'{LEFT_DATA_DIR}' フォルダからデータを読み込み中...")
        left_actions = blosc2.load_array(os.path.join(LEFT_DATA_DIR, "action_states.blosc2"))
        left_states = blosc2.load_array(os.path.join(LEFT_DATA_DIR, "joint_states.blosc2"))
        
        # images.blosc2 が存在するか確認
        left_images_path = os.path.join(LEFT_DATA_DIR, "image_states.blosc2")
        has_images = os.path.exists(left_images_path)
        if has_images:
            left_images = blosc2.load_array(left_images_path)

        print(f"'{RIGHT_DATA_DIR}' フォルダからデータを読み込み中...")
        right_actions = blosc2.load_array(os.path.join(RIGHT_DATA_DIR, "action_states.blosc2"))
        right_states = blosc2.load_array(os.path.join(RIGHT_DATA_DIR, "joint_states.blosc2"))

        if has_images:
            right_images_path = os.path.join(RIGHT_DATA_DIR, "image_states.blosc2")
            if os.path.exists(right_images_path):
                right_images = blosc2.load_array(right_images_path)
            else:
                # もしrightデータにimagesがなければ、空の配列で埋める
                print(f"警告: '{RIGHT_DATA_DIR}' に 'images_states.blosc2' が見つかりません。ゼロで埋めます。")
                img_shape = list(left_images.shape)
                img_shape[0] = right_actions.shape[0]
                right_images = np.zeros(tuple(img_shape), dtype=left_images.dtype)
        
        # データが40個以上あるかチェック
        if len(left_actions) < NUM_EPISODES_EACH or len(right_actions) < NUM_EPISODES_EACH:
            print(f"エラー: データの数が{NUM_EPISODES_EACH}個未満です。")
            return

        # --- データを交互に混ぜる ---
        merged_actions = []
        merged_states = []
        if has_images:
            merged_images = []

        print(f"データを交互に混ぜています (各{NUM_EPISODES_EACH}エピソード)...")
        for i in range(NUM_EPISODES_EACH):
            # 最初に右のデータ
            merged_actions.append(right_actions[i])
            merged_states.append(right_states[i])
            if has_images:
                merged_images.append(right_images[i])
            
            # 次に左のデータ
            merged_actions.append(left_actions[i])
            merged_states.append(left_states[i])
            if has_images:
                merged_images.append(left_images[i])

        # --- 新しいデータとして保存する ---
        final_actions = np.array(merged_actions)
        final_states = np.array(merged_states)
        if has_images:
            final_images = np.array(merged_images)

        # 出力用のフォルダを作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"新しいフォルダ '{OUTPUT_DIR}' に書き込み中...")

        blosc2.save_array(final_actions, os.path.join(OUTPUT_DIR, "actions.blosc2"))
        blosc2.save_array(final_states, os.path.join(OUTPUT_DIR, "states.blosc2"))
        if has_images:
            blosc2.save_array(final_images, os.path.join(OUTPUT_DIR, "images.blosc2"))

        print("\nマージが完了しました！")
        print("--- 作成されたデータの情報 ---")
        print(f"合計エピソード数: {len(final_actions)}")
        print(f"Actions shape: {final_actions.shape}")
        print(f"States shape: {final_states.shape}")
        if has_images:
            print(f"Images shape: {final_images.shape}")
        print("--------------------------")

    except FileNotFoundError as e:
        print(f"エラー: ファイルまたはフォルダが見つかりません。パスを確認してください: {e.filename}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    merge_datasets()