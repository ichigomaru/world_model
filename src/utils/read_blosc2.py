import blosc2
import numpy as np

def load_blosc2(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return blosc2.unpack_array2(f.read())
    # r読み取り。bバイナリ(圧縮ファイルなど)
    # blosc2.unpack_array2(...)はblosc2が提供しているライブラリで、圧縮されたデータの復元