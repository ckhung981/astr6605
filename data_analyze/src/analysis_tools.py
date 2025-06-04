# src/analysis_tools.py (CGS 單位版)
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

    def plot_energy_evolution(self, snapshot_numbers, kinetic_energies, potential_energies):
        """
        繪製動能對時間、位能對時間、總能量對時間、維里比對時間的圖表。
        能量單位為 erg。
        
        Args:
            snapshot_numbers (list): 快照編號列表 (時間軸)。
            kinetic_energies (list): 每個快照的總動能列表，單位 erg。
            potential_energies (list): 每個快照的總位能列表，單位 erg。
        """
        if not snapshot_numbers:
            print(f"No energy data to plot for {self.sim_name}.")
            return

        snapshot_numbers_arr = np.array(snapshot_numbers)
        kinetic_energies_arr = np.array(kinetic_energies)
        potential_energies_arr = np.array(potential_energies)
        
        total_energies_arr = kinetic_energies_arr + potential_energies_arr
        virial_quantities_arr = 2 * kinetic_energies_arr + potential_energies_arr

        # --- 繪製動能對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, kinetic_energies_arr, marker='o', linestyle='-')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Kinetic Energy [erg]') # 標籤更新為 erg
        plt.title(f'{self.sim_name} - Kinetic Energy Evolution')
        plt.grid(True)
        ke_filename = os.path.join(self.output_dir, f'{self.sim_name}_kinetic_energy_evolution.png')
        plt.savefig(ke_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {ke_filename}")

        # --- 繪製位能對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, potential_energies_arr, marker='o', linestyle='-', color='red')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Potential Energy [erg]') # 標籤更新為 erg
        plt.title(f'{self.sim_name} - Potential Energy Evolution')
        plt.grid(True)
        pe_filename = os.path.join(self.output_dir, f'{self.sim_name}_potential_energy_evolution.png')
        plt.savefig(pe_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {pe_filename}")
        
        # --- 繪製總能量對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, total_energies_arr, marker='o', linestyle='-', color='purple')
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('Total Energy (T + U) [erg]') # 標籤更新為 erg
        plt.title(f'{self.sim_name} - Total Energy Evolution')
        plt.grid(True)
        te_filename = os.path.join(self.output_dir, f'{self.sim_name}_total_energy_evolution.png')
        plt.savefig(te_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {te_filename}")

        # --- 繪製維里定理 (2T + U) 對時間圖 ---
        plt.figure(figsize=(10, 6))
        plt.plot(snapshot_numbers_arr, virial_quantities_arr, marker='o', linestyle='-', color='green')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Virial Equilibrium (2T+U=0)') 
        plt.xlabel('Snapshot Number (Time)')
        plt.ylabel('2 * Kinetic Energy + Potential Energy [erg]') # 標籤更新為 erg
        plt.title(f'{self.sim_name} - Virial Theorem Check (2T + U) Evolution')
        plt.legend()
        plt.grid(True)
        virial_filename = os.path.join(self.output_dir, f'{self.sim_name}_virial_theorem_evolution.png')
        plt.savefig(virial_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {virial_filename}")