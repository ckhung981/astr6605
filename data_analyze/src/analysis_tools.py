# src/analysis_tools.py (CGS 單位版，新增累積位力演化圖)
import matplotlib.pyplot as plt
import os
import numpy as np

class AnalysisTools:
    """
    負責對模擬數據進行分析並繪製時間演化圖表。
    所有能量數據都假定為 CGS 單位 (erg)。
    """
    def __init__(self, output_dir, sim_name):
        self.output_dir = output_dir
        self.sim_name = sim_name

    def plot_energy_evolution(self, snapshot_numbers, kinetic_energies, potential_energies, filename_suffix=""):
        """
        繪製動能對時間、位能對時間、總能量對時間、維里比對時間的圖表。
        能量單位為 erg。
        
        Args:
            snapshot_numbers (list or np.ndarray): 快照編號列表 (時間軸)。
            kinetic_energies (list or np.ndarray): 每個快照的總動能列表，單位 erg。
            potential_energies (list or np.ndarray): 每個快照的總位能列表，單位 erg。
            filename_suffix (str): 添加到輸出圖片檔名上的後綴。
        """
        # 確保數據是 NumPy 陣列
        snapshot_numbers_arr = np.array(snapshot_numbers)
        kinetic_energies_arr = np.array(kinetic_energies)
        potential_energies_arr = np.array(potential_energies)

        # 修正空陣列檢查：使用 .size 屬性
        if snapshot_numbers_arr.size == 0:
            print(f"No energy data to plot for {self.sim_name} with suffix '{filename_suffix}'. Array is empty after filtering/slicing.")
            return

        total_energies_arr = kinetic_energies_arr + potential_energies_arr
        virial_quantities_arr = 2 * kinetic_energies_arr + potential_energies_arr

        # --- 繪製動能對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, kinetic_energies_arr, marker='o', linestyle='-')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Kinetic Energy [erg]')
        plt.title(f'{self.sim_name} - Kinetic Energy Evolution')
        plt.grid(True)
        ke_filename = os.path.join(self.output_dir, f'{self.sim_name}_kinetic_energy_evolution{filename_suffix}.png')
        plt.savefig(ke_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {ke_filename}")

        # --- 繪製位能對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, potential_energies_arr, marker='o', linestyle='-', color='red')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Potential Energy [erg]')
        plt.title(f'{self.sim_name} - Potential Energy Evolution')
        plt.grid(True)
        pe_filename = os.path.join(self.output_dir, f'{self.sim_name}_potential_energy_evolution{filename_suffix}.png')
        plt.savefig(pe_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {pe_filename}")
        
        # --- 繪製總能量對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, total_energies_arr, marker='o', linestyle='-', color='purple')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Total Energy (T + U) [erg]')
        plt.title(f'{self.sim_name} - Total Energy Evolution')
        plt.grid(True)
        te_filename = os.path.join(self.output_dir, f'{self.sim_name}_total_energy_evolution{filename_suffix}.png')
        plt.savefig(te_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {te_filename}")

        # --- 繪製維里定理 (2T + U) 對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, virial_quantities_arr, marker='o', linestyle='-', color='green')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Virial Equilibrium (2T+U=0)') 
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('2 * Kinetic Energy + Potential Energy [erg]')
        plt.title(f'{self.sim_name} - Virial Theorem Check (2T + U) Evolution')
        plt.legend()
        plt.grid(True)
        virial_filename = os.path.join(self.output_dir, f'{self.sim_name}_virial_theorem_evolution{filename_suffix}.png')
        plt.savefig(virial_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {virial_filename}")

    def plot_cumulative_virial_evolution(self, snapshot_numbers, kinetic_energies, potential_energies, filename_suffix=""):
        """
        計算並繪製累積平均動能和位能的位力定理關係 (2<T> + <U>) 隨時間的變化。
        
        Args:
            snapshot_numbers (np.ndarray): 快照編號陣列。
            kinetic_energies (np.ndarray): 動能陣列。
            potential_energies (np.ndarray): 位能陣列。
            filename_suffix (str): 添加到輸出圖片檔名上的後綴。
        """
        # 確保數據是 NumPy 陣列
        snapshot_numbers_arr = np.array(snapshot_numbers)
        kinetic_energies_arr = np.array(kinetic_energies)
        potential_energies_arr = np.array(potential_energies)

        if snapshot_numbers_arr.size == 0:
            print(f"No data to plot cumulative virial for {self.sim_name} with suffix '{filename_suffix}'. Array is empty.")
            return

        # 計算累積平均動能和位能
        cumulative_kinetic_avg = np.zeros_like(kinetic_energies_arr, dtype=float)
        cumulative_potential_avg = np.zeros_like(potential_energies_arr, dtype=float)
       
        for i in range(snapshot_numbers_arr.size):
            cumulative_kinetic_avg[i] = np.mean(kinetic_energies_arr[:i+1])
            cumulative_potential_avg[i] = np.mean(potential_energies_arr[:i+1])
        print(cumulative_kinetic_avg)
        print(cumulative_potential_avg)
        # 計算累積平均的 2<T> + <U>
        cumulative_virial_quantity = 2 * cumulative_kinetic_avg + cumulative_potential_avg
        print(cumulative_virial_quantity)
        # --- 繪製累積平均 2<T> + <U> 對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, cumulative_virial_quantity, marker='o', linestyle='-', color='blue')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Virial Equilibrium (2<T>+<U>)') 
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('2 * <Kinetic Energy> + <Potential Energy> [erg]')
        plt.title(f'{self.sim_name} - Virial Theorem Check')
        plt.legend()
        plt.grid(True)
        
        cumulative_virial_filename = os.path.join(self.output_dir, f'{self.sim_name}_cumulative_virial_evolution{filename_suffix}.png')
        plt.savefig(cumulative_virial_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {cumulative_virial_filename}")