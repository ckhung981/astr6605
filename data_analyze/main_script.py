import os
import numpy as np
import h5py

# 從 src 資料夾導入模組，並給予簡潔的別名
import src.utils as utils
import src.data_reader as dr
import src.plotting_tools as pt
import src.video_creator as vc
import src.analysis_tools as at 

def _generate_output_subdir_name(sample_size, num_snapshots, min_dist_kpc):
    """
    根據關鍵參數生成一個簡潔的子資料夾名稱。
    例如: s100k_n060_d0p1
    """
    # 縮寫 particle_sample_size
    if sample_size >= 1_000_000:
        s_str = f"s{int(sample_size/1_000_000)}M"
    elif sample_size >= 1_000:
        s_str = f"s{int(sample_size/1_000)}k"
    else:
        s_str = f"s{sample_size}"
    
    # 縮寫 num_snapshots_to_process
    n_str = f"n{num_snapshots:03d}" # 確保是三位數，例如 n060
    
    # 縮寫 min_distance_from_origin_for_sampling_kpc
    # 將小數點替換為 'p'，並處理零
    if min_dist_kpc == 0:
        d_str = "d0"
    else:
        # 格式化為一位小數，並將小數點替換為 'p'
        d_str = f"d{min_dist_kpc:.1f}".replace('.', 'p') 
    
    return f"{s_str}_{n_str}_{d_str}"


def main():
    """
    主執行函數，處理數據讀取、繪圖和影片生成。
    所有物理量現在都將轉換並使用 CGS 單位 (cm, g, s, erg)。
    """
    # --- 配置參數 ---
    parent_data_base_dir = '/data/astr6605/'
    base_output_dir = '/data/astr6605/data_analyze/result/' # 這裡是主輸出目錄 
    
    snapshot_pattern = 'snapshot_{:03d}.hdf5'
    num_snapshots_to_process = 129 # 處理快照 000 到 059 (共 60 個)
    
    # *** 關鍵參數：粒子採樣大小和位能計算方法 ***
    # 請根據您的需求和計算能力調整這些參數。
    # 如果要使用 O(N^2) 精確位能計算並觀察維里定理，建議將 particle_sample_size 設置為較小的值 (例如 1000)。
    particle_sample_size = 100000 
    
    # 位能計算方法：
    # 'direct': O(N^2) 精確計算所有粒子對的位能。物理上符合維里定理的 U。
    #           如果 `particle_sample_size` > `max_particles_for_pe_direct_calc`，
    #           則 `dr.calculate_potential_energy` 會返回 0.0。
    # 'com_approx': 假設質量中心在原點 (0,0,0)，使用 O(N) 簡化近似。
    #               適合大量粒子，但位能是近似值，不嚴格符合維里定理的 U。
    potential_energy_calculation_method = 'direct' 
    
    # 如果 potential_energy_calculation_method == 'direct'，此參數生效
    # 此參數現在主要作為一個“推薦上限”或“警告閾值”，不再強制返回0。
    # 若要計算 100,000 粒子的 O(N^2) 位能，即使有 Numba 也會非常慢 (大約 $10^{10}$ 次操作)。
    max_particles_for_pe_direct_calc = particle_sample_size 
                                            
    video_fps = 5
    
    # 萬有引力常數 G，現在設定為 CGS 單位：cm^3 g^-1 s^-2
    gravitational_constant_G = dr.HDF5DataReader.G_CGS # 從 data_reader 模組獲取 CGS 單位 G 值

    # --- 粒子採樣時，距離原點的最小距離 ---
    # 這裡的設定單位是 kpc，程式會在傳遞給 data_reader 前轉換為 cm。
    min_distance_from_origin_for_sampling_kpc = 0.0 # kpc

    # 確保主輸出資料夾存在
    utils.ensure_directory_exists(base_output_dir)

    # --- 指定要處理的模擬資料夾名稱 ---
    simulation_dirs_to_process = ['gravity_6'] 
    # simulation_dirs_to_process = ['gravity', 'gravity_2'] 
    # simulation_dirs_to_process = utils.get_simulation_dirs(parent_data_base_dir, prefix='gravity')


    # 檢查指定的資料夾是否存在於 parent_data_base_dir 下
    valid_simulation_dirs = []
    for sim_name in simulation_dirs_to_process:
        full_path = os.path.join(parent_data_base_dir, sim_name, 'output')
        if os.path.exists(full_path) and os.path.isdir(full_path):
            valid_simulation_dirs.append(sim_name)
        else:
            print(f"Warning: Specified simulation directory '{full_path}' not found or not an output directory. Skipping.")

    simulation_dirs = valid_simulation_dirs

    if not simulation_dirs:
        print(f"Error: No valid simulation directories found among {simulation_dirs_to_process}. Exiting.")
        return

    # --- 第一階段：計算全局速度範圍 (CGS) 和座標範圍 (CGS) (用於統一 Colorbar 和圖軸) ---
    global_min_vel_cgs = float('inf')
    global_max_vel_cgs = float('-inf')
    global_abs_max_coord_cgs = 0.0 # 用於儲存所有快照中座標的最大絕對值 (CGS)

    print("\n" + "="*80)
    print("--- Phase 1: Calculating global min/max velocities and coordinate ranges (CGS) ---")
    print("="*80 + "\n")

    for sim_dir_name in simulation_dirs:
        data_dir = os.path.join(parent_data_base_dir, sim_dir_name, 'output')
        print(f"  Scanning {sim_dir_name} for ranges...")
        for snap_num in range(num_snapshots_to_process):
            file_path = os.path.join(data_dir, snapshot_pattern.format(snap_num))
            if not os.path.exists(file_path):
                continue
            
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'PartType1' not in f:
                        continue
                    p_type_group = f['PartType1']

                    unit_length_in_cm = f['Parameters'].attrs.get('UnitLength_in_cm', 1.0)
                    unit_velocity_in_cm_per_s = f['Parameters'].attrs.get('UnitVelocity_in_cm_per_s', 1.0)

                    # 讀取速度 (轉換為 CGS)
                    if 'Velocities' in p_type_group:
                        velocities_internal = p_type_group['Velocities'][:]
                        velocities_cgs_current = velocities_internal * unit_velocity_in_cm_per_s 
                        
                        vel_magnitude_cgs = np.sqrt(np.sum(velocities_cgs_current**2, axis=1))
                        if len(vel_magnitude_cgs) > 0:
                            global_min_vel_cgs = min(global_min_vel_cgs, np.min(vel_magnitude_cgs))
                            global_max_vel_cgs = max(global_max_vel_cgs, np.max(vel_magnitude_cgs))
                    
                    # 讀取座標並更新全局範圍 (轉換為 CGS)
                    if 'Coordinates' in p_type_group:
                        coordinates_internal = p_type_group['Coordinates'][:] 
                        coordinates_cgs_current = coordinates_internal * unit_length_in_cm
                        
                        # --- Check for NaN/Inf in coordinates before using them for global_abs_max_coord_cgs ---
                        if np.any(np.isnan(coordinates_cgs_current)) or np.any(np.isinf(coordinates_cgs_current)):
                            print(f"    Warning: Skipping coordinates in {file_path} for global range calculation due to NaN/Inf values.")
                            continue # Skip this file for global range calculation
                        
                        if len(coordinates_cgs_current) > 0:
                            current_max_abs_coord_cgs = np.max(np.abs(coordinates_cgs_current))
                            global_abs_max_coord_cgs = max(global_abs_max_coord_cgs, current_max_abs_coord_cgs)

            except (FileNotFoundError, KeyError, Exception) as e:
                print(f"    Warning: Skipping {file_path} for range calculation due to error: {e}")
                continue

    # 處理速度範圍的邊界情況和緩衝 (CGS)
    if global_min_vel_cgs == float('inf') or global_max_vel_cgs == float('-inf'):
        print("No valid velocity data found across all snapshots. Using default colorbar range (0.0 to 1.0e7 cm/s).")
        global_min_vel_cgs = 0.0
        global_max_vel_cgs = 1.0e7 # 100 km/s in cm/s for default
    elif global_min_vel_cgs == global_max_vel_cgs:
        global_min_vel_cgs = max(0.0, global_min_vel_cgs - 1.0)
        global_max_vel_cgs += 1.0
    else:
        range_buffer_vel = (global_max_vel_cgs - global_min_vel_cgs) * 0.05
        global_min_vel_cgs = max(0.0, global_min_vel_cgs - range_buffer_vel)
        global_max_vel_cgs += range_buffer_vel

    # 確定最終的座標繪圖範圍 (CGS)，可以加入一些緩衝
    # --- Robustness check for global_abs_max_coord_cgs ---
    if global_abs_max_coord_cgs <= 1e-10: # Check if effectively zero, use a small threshold
        print("Warning: Global maximum coordinate is effectively zero. Using default plot range (-1 kpc to 1 kpc in cm).")
        global_plot_range_cgs = (-dr.HDF5DataReader.KPC_TO_CM, dr.HDF5DataReader.KPC_TO_CM) 
    else:
        coord_buffer_cgs = global_abs_max_coord_cgs * 0.1 # 10% 緩衝
        global_plot_range_cgs = (-global_abs_max_coord_cgs - coord_buffer_cgs, global_abs_max_coord_cgs + coord_buffer_cgs)

    print(f"\nGlobal Velocity Range (for Colorbar): Min = {global_min_vel_cgs:.2e} cm/s, Max = {global_max_vel_cgs:.2e} cm/s")
    print(f"Global Coordinate Plot Range: X/Y/Z from {global_plot_range_cgs[0]:.2e} to {global_plot_range_cgs[1]:.2e} cm")
    print("\n" + "="*80 + "\n")


    # --- 第二階段：處理每個模擬資料夾，生成圖片、影片和能量圖 ---
    print("\n" + "="*80)
    print("--- Phase 2: Processing each simulation and generating plots/videos/energy charts ---")
    print("="*80 + "\n")
    video_creator = vc.VideoCreator(fps=video_fps)

    for sim_dir_name in simulation_dirs:
        # 生成包含參數的子資料夾名稱
        param_subdir_name = _generate_output_subdir_name(
            particle_sample_size, 
            num_snapshots_to_process, 
            min_distance_from_origin_for_sampling_kpc
        )
        # 更新 current_output_dir 以包含新的參數子資料夾
        current_output_dir = os.path.join(base_output_dir, sim_dir_name, param_subdir_name)
        utils.ensure_directory_exists(current_output_dir) # 確保該模擬的輸出資料夾存在
        
        data_dir = os.path.join(parent_data_base_dir, sim_dir_name, 'output') # 原始數據來源路徑
        
        print(f"\nProcessing simulation: {sim_dir_name} with parameters: {param_subdir_name}")
        print(f"  Saving plots to: {current_output_dir}")

        # 每個模擬的能量數據檔案路徑
        energy_data_filepath = os.path.join(current_output_dir, f'{sim_dir_name}_energy_data.npy')
        
        # 檢查是否已存在能量數據檔案
        if os.path.exists(energy_data_filepath):
            print(f"  Info: Found existing energy data at {energy_data_filepath}. Loading it instead of re-calculating.")
            try:
                loaded_data = np.load(energy_data_filepath, allow_pickle=True).item()
                snapshot_numbers_for_energy = loaded_data['snapshot_numbers']
                kinetic_energies = loaded_data['kinetic_energies']
                potential_energies = loaded_data['potential_energies']
                
                # 如果要重新生成所有圖片，則繼續執行繪圖迴圈
                # 如果只是要繪製能量圖，則可以跳過下面的 for snap_num 迴圈
                # For now, we will still generate plots, assuming energy data is faster than plotting.
                # If plotting is also slow, you might want a separate flag to skip it.
                
                # For video generation, we still need to generate the individual images
                # unless you save them and check for their existence too.
                # For now, we'll assume image generation is always desired.

            except Exception as e:
                print(f"  Warning: Error loading energy data: {e}. Re-calculating energy data.")
                snapshot_numbers_for_energy = []
                kinetic_energies = []
                potential_energies = []
        else:
            print(f"  Info: No existing energy data found. Calculating energies for all snapshots.")
            snapshot_numbers_for_energy = []
            kinetic_energies = []
            potential_energies = []


        xy_filenames = []
        xz_filenames = []
        yz_filenames = []
        
        plotting_tools = pt.PlottingTools(current_output_dir, sim_dir_name, 
                                          global_vel_range=(global_min_vel_cgs, global_max_vel_cgs),
                                          global_coord_range=global_plot_range_cgs) 
        analysis_tools = at.AnalysisTools(current_output_dir, sim_dir_name)

        min_distance_from_origin_for_sampling_cm = min_distance_from_origin_for_sampling_kpc * dr.HDF5DataReader.KPC_TO_CM

        # 只有在沒有載入能量數據的情況下才進行計算，否則直接跳到繪圖
        if not snapshot_numbers_for_energy or len(snapshot_numbers_for_energy) == 0:
            for snap_num in range(num_snapshots_to_process):
                file_path = os.path.join(data_dir, snapshot_pattern.format(snap_num))
                
                if not os.path.exists(file_path):
                    print(f"  Warning: {file_path} not found, skipping snapshot {snap_num:03d}...")
                    continue
                
                print(f"  Processing snapshot: {snap_num:03d} from {file_path}...")
                
                try:
                    reader = dr.HDF5DataReader(file_path)
                    reader.process_data(
                        sample_size=particle_sample_size,
                        min_dist_from_origin=min_distance_from_origin_for_sampling_cm 
                    ) 

                    coordinates = reader.get_coordinates() 
                    vel_magnitude = reader.get_velocity_magnitude() 
                    
                    ke = reader.calculate_kinetic_energy() 
                    
                    pe = reader.calculate_potential_energy(
                        G=gravitational_constant_G, 
                        method=potential_energy_calculation_method, 
                        max_particles_for_direct=max_particles_for_pe_direct_calc 
                    )
                    
                    print(f"    Snapshot {snap_num:03d} (Particles: {len(coordinates) if coordinates is not None else 0}/{reader.get_num_particles_read()}): KE = {ke:.2e} erg, PE = {pe:.2e} erg")
                    
                    if coordinates is not None and len(coordinates) > 0:
                        snapshot_numbers_for_energy.append(snap_num)
                        kinetic_energies.append(ke)
                        potential_energies.append(pe)
                        
                        # Always generate projection plots if there are valid particles
                        saved_files = plotting_tools.plot_2d_projection(coordinates, vel_magnitude, snap_num)
                        xy_filenames.append(saved_files['XY'])
                        xz_filenames.append(saved_files['XZ'])
                        yz_filenames.append(saved_files['YZ'])
                        print(f"  Saved projections for snapshot {snap_num:03d}")
                    else:
                        print(f"  No valid particle data found for snapshot {snap_num:03d}. Skipping plotting and energy calculation.")

                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
                    continue
            
            # --- 保存能量數據 ---
            if snapshot_numbers_for_energy:
                energy_data = {
                    'snapshot_numbers': np.array(snapshot_numbers_for_energy),
                    'kinetic_energies': np.array(kinetic_energies),
                    'potential_energies': np.array(potential_energies)
                }
                np.save(energy_data_filepath, energy_data)
                print(f"  Info: Energy data saved to {energy_data_filepath}")
            else:
                print(f"  Info: No energy data to save for {sim_dir_name} parameters: {param_subdir_name}.")
        else:
            # If energy data was loaded, we still need to generate plots for video creation
            # This part needs to re-read data for plotting even if energy is loaded.
            # This is a trade-off. If plotting is slow, consider adding a flag to skip plotting.
            print(f"  Info: Energy data loaded. Re-generating projection plots for video (if needed).")
            for snap_num in snapshot_numbers_for_energy: # Loop through available snapshots
                file_path = os.path.join(data_dir, snapshot_pattern.format(snap_num))
                if not os.path.exists(file_path):
                    print(f"  Warning: {file_path} not found for plotting, skipping.")
                    continue
                try:
                    reader = dr.HDF5DataReader(file_path)
                    reader.process_data(
                        sample_size=particle_sample_size,
                        min_dist_from_origin=min_distance_from_origin_for_sampling_cm
                    )
                    coordinates = reader.get_coordinates()
                    vel_magnitude = reader.get_velocity_magnitude()
                    if coordinates is not None and len(coordinates) > 0:
                        saved_files = plotting_tools.plot_2d_projection(coordinates, vel_magnitude, snap_num)
                        xy_filenames.append(saved_files['XY'])
                        xz_filenames.append(saved_files['XZ'])
                        yz_filenames.append(saved_files['YZ'])
                    else:
                        print(f"  No valid particle data for snapshot {snap_num:03d} for plotting.")
                except Exception as e:
                    print(f"  Error generating plot for {file_path}: {e}")

        # --- 影片生成 (每個模擬資料夾完成後) ---
        print(f"\n  Generating videos for {sim_dir_name} parameters: {param_subdir_name}...")
        video_creator.create_video_from_images(
            xy_filenames, 
            os.path.join(current_output_dir, f'{sim_dir_name}_XY_projection.mp4'), 
            "XY Projection"
        )
        video_creator.create_video_from_images(
            xz_filenames, 
            os.path.join(current_output_dir, f'{sim_name}_XZ_projection.mp4'), 
            "XZ Projection" 
        )
        video_creator.create_video_from_images(
            yz_filenames, 
            os.path.join(current_output_dir, f'{sim_name}_YZ_projection.mp4'), 
            "YZ Projection" 
        )

        # --- 能量演化圖表生成 (每個模擬資料夾完成後) ---
        print(f"  Generating energy evolution plots for {sim_dir_name} parameters: {param_subdir_name}...")
        analysis_tools.plot_energy_evolution(
            snapshot_numbers_for_energy, 
            kinetic_energies, 
            potential_energies
        )

    print("\n" + "="*80)
    print("--- All processing complete! ---")
    print("="*80 + "\n")

# 當此腳本作為主程式運行時，執行 main 函數
if __name__ == '__main__':
    main()

    # 如果需要測試 HDF5DataReader 的結構打印功能，可以這樣調用：
    # print("\n--- Testing HDF5 structure print function ---")
    # dr.HDF5DataReader.print_structure('/data/astr6605/gravity/output/snapshot_000.hdf5')