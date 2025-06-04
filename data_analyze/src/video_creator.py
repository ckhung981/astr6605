import imageio
import os

class VideoCreator:
    """
    負責將一系列圖片轉換為影片。
    """
    def __init__(self, fps=5):
        self.fps = fps

    def create_video_from_images(self, image_filenames, output_video_path, description=""):
        """
        將一系列圖片按照順序組合成一個 MP4 影片。
        
        Args:
            image_filenames (list): 包含所有圖片檔案路徑的列表。
            output_video_path (str): 輸出影片的完整路徑和檔名。
            description (str): 影片的簡短描述，用於日誌輸出。
        """
        if not image_filenames:
            print(f"Warning: No images provided for {description} video creation. Skipping.")
            return

        print(f"Creating {description} video: {output_video_path}")
        try:
            # 確保圖片按照檔名（通常是快照編號）排序
            sorted_filenames = sorted(image_filenames)
            
            with imageio.get_writer(output_video_path, fps=self.fps) as writer:
                for filename in sorted_filenames:
                    if os.path.exists(filename):
                        image = imageio.imread(filename)
                        writer.append_data(image)
                    else:
                        print(f"  Warning: Image file not found, skipping: {filename}")
            print(f"Successfully created {output_video_path}")
        except Exception as e:
            print(f"Error creating {description} video ({output_video_path}): {e}")