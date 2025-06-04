# src/plotting_tools.py (CGS 單位版)
import matplotlib.pyplot as plt
import numpy as np
import os

class PlottingTools:
    """
    負責生成模擬數據的 2D 投影圖。
    所有輸入數據 (座標、速度) 都假定為 CGS 單位。
    """
    def __init__(self, output_dir, sim_name, global_vel_range=(0.0, 1.0e7), global_coord_range=(-3.086e21, 3.086e21)):
        self.output_dir = output_dir
        self.sim_name = sim_name
        self.global_min_vel, self.global_max_vel = global_vel_range
        self.global_min_coord, self.global_max_coord = global_coord_range

    def plot_2d_projection(self, coordinates, vel_magnitude, snap_num):
        """
        為給定快照數據生成 XY, XZ, YZ 投影圖。
        輸入座標為 cm，速度為 cm/s。
        
        Args:
            coordinates (np.ndarray): 粒子坐標 (N, 3)，單位 cm。
            vel_magnitude (np.ndarray): 粒子速度幅度 (N,)，單位 cm/s。
            snap_num (int): 快照編號。
        
        Returns:
            dict: 包含各投影圖儲存路徑的字典。
        """
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]

        projections_info = [
            ('XY', x, y, 'X [cm]', 'Y [cm]'), # 座標軸標籤更新為 cm
            ('XZ', x, z, 'X [cm]', 'Z [cm]'),
            ('YZ', y, z, 'Y [cm]', 'Z [cm]')
        ]

        saved_filenames = {}

        for plane, x_data, y_data, x_label, y_label in projections_info:
            filename = os.path.join(self.output_dir, f'{plane.lower()}_projection_{snap_num:03d}.png')
            
            fig, ax = plt.subplots(figsize=(8, 6)) 
            
            scatter = ax.scatter(x_data, y_data, 
                                  c=vel_magnitude, 
                                  cmap='viridis', 
                                  s=0.5,           
                                  alpha=1.0,       
                                  vmin=self.global_min_vel, 
                                  vmax=self.global_max_vel)
            
            # Colorbar 標籤更新為 cm/s
            fig.colorbar(scatter, ax=ax, label='Velocity Magnitude [cm/s]') 
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{self.sim_name} - {plane} Projection (Snapshot {snap_num:03d})')
            
            # 設置固定的座標範圍 (CGS)
            ax.set_xlim(self.global_min_coord, self.global_max_coord)
            ax.set_ylim(self.global_min_coord, self.global_max_coord)
            ax.set_aspect('equal', adjustable='box') 
            
            fig.set_tight_layout(True) 
            
            fig.savefig(filename, dpi=300) 
            plt.close(fig) 
            
            saved_filenames[plane] = filename
        
        return saved_filenames