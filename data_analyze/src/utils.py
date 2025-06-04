import os

def ensure_directory_exists(path):
    """
    確保指定路徑的資料夾存在，如果不存在則創建。
    """
    os.makedirs(path, exist_ok=True)
    print(f"Directory ensured: {path}")

def get_simulation_dirs(parent_base_dir, prefix='gravity'):
    """
    在指定的父資料夾中找到所有以 'gravity' 開頭的模擬資料夾。
    """
    return [d for d in os.listdir(parent_base_dir) 
            if os.path.isdir(os.path.join(parent_base_dir, d)) and d.startswith(prefix)]