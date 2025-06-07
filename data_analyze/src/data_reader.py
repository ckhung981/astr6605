# src/data_reader.py (移除位能計算粒子數量限制)
import h5py
import numpy as np
import os
import numba # 導入 numba

class HDF5DataReader:
    """
    負責從 HDF5 快照檔案中讀取原始數據，並將其轉換為 CGS 單位。
    """
    SOLAR_MASS_IN_GRAMS = 1.989e33 # 1 太陽質量 (M_sun) = 1.989e33 克
    KPC_TO_CM = 3.086e21 # 1 kpc = 3.086e+21 cm
    KM_PER_S_TO_CM_PER_S = 1.0e5 # 1 km/s = 1.0e5 cm/s
    G_CGS = 6.67e-8 # 萬有引力常數 G，單位 cm^3 g^-1 s^-2

    def __init__(self, file_path, particle_type='PartType1'):
        self.file_path = file_path
        self.particle_type = particle_type
        
        self.coordinates_internal = None
        self.velocities_internal = None
        self.masses_internal = None

        self.coordinates_cgs = None # CGS 單位
        self.velocities_cgs = None  # CGS 單位
        self.masses_cgs = None      # CGS 單位

        self.vel_magnitude_cgs = None # CGS 單位速度幅度
        self.num_particles_read = 0

        self.unit_length_in_cm = None
        self.unit_mass_in_g = None
        self.unit_velocity_in_cm_per_s = None

    def _read_raw_data_and_units(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with h5py.File(self.file_path, 'r') as f:
            if 'Parameters' in f.keys():
                params = f['Parameters'].attrs
                self.unit_length_in_cm = params.get('UnitLength_in_cm', 1.0)
                self.unit_mass_in_g = params.get('UnitMass_in_g', 1.0)
                self.unit_velocity_in_cm_per_s = params.get('UnitVelocity_in_cm_per_s', 1.0)
            else:
                print("Warning: 'Parameters' group not found. Assuming unit conversion factors are 1.0.")
                self.unit_length_in_cm = 1.0
                self.unit_mass_in_g = 1.0
                self.unit_velocity_in_cm_per_s = 1.0
            
            if 'Header' not in f:
                raise KeyError("Header group not found in HDF5 file.")
            if 'MassTable' not in f['Header'].attrs:
                raise KeyError("MassTable attribute not found in Header.")
            if 'NumPart_ThisFile' not in f['Header'].attrs:
                raise KeyError("NumPart_ThisFile attribute not found in Header.")

            if self.particle_type not in f:
                raise KeyError(f"Particle type '{self.particle_type}' not found in {self.file_path}")
            
            p_type_group = f[self.particle_type]

            if 'Coordinates' not in p_type_group:
                raise KeyError(f"'Coordinates' dataset not found under '{self.particle_type}' in {self.file_path}")
            if 'Velocities' not in p_type_group:
                raise KeyError(f"'Velocities' dataset not found under '{self.particle_type}' in {self.file_path}")
            
            self.coordinates_internal = p_type_group['Coordinates'][:]
            self.velocities_internal = p_type_group['Velocities'][:]

            particle_type_index = int(self.particle_type.replace('PartType', ''))
            mass_table = f['Header'].attrs['MassTable']
            num_part_this_file = f['Header'].attrs['NumPart_ThisFile']

            num_particles_of_this_type = num_part_this_file[particle_type_index]

            if num_particles_of_this_type == 0:
                self.masses_internal = np.array([])
                self.coordinates_internal = np.array([])
                self.velocities_internal = np.array([])
                self.num_particles_read = 0
                print(f"  Info: No particles of {self.particle_type} in this file. Skipping raw data reading.")
                return 

            if mass_table[particle_type_index] != 0:
                self.masses_internal = np.full(num_particles_of_this_type, mass_table[particle_type_index], dtype=np.float64)
                print(f"  Info: Reading Masses from Header/MassTable for {self.particle_type}: {mass_table[particle_type_index]:.2e} (internal units)")
            elif 'Masses' in p_type_group:
                self.masses_internal = p_type_group['Masses'][:]
                print(f"  Info: Reading Masses from '{self.particle_type}/Masses' dataset (internal units).")
            else:
                print(f"Warning: No explicit mass for {self.particle_type} in MassTable or as a 'Masses' dataset. Assuming unit mass (1.0 internal unit) for calculations.")
                self.masses_internal = np.ones(num_particles_of_this_type, dtype=np.float64)
            
            self.num_particles_read = len(self.coordinates_internal)

    def process_data(self, sample_size=100000, min_dist_from_origin=0.0):
        self._read_raw_data_and_units()

        if self.num_particles_read == 0:
            self.coordinates_cgs = np.array([])
            self.velocities_cgs = np.array([])
            self.masses_cgs = np.array([])
            self.vel_magnitude_cgs = np.array([])
            return

        self.coordinates_cgs = self.coordinates_internal * self.unit_length_in_cm  # 單位：cm
        self.velocities_cgs = self.velocities_internal * self.unit_velocity_in_cm_per_s # 單位：cm/s
        self.masses_cgs = self.masses_internal * self.unit_mass_in_g # 單位：g
        
        if len(self.masses_cgs) > 0:
            print(f"  Converted mass example: {self.masses_internal[0]:.2e} (internal) -> {self.masses_cgs[0]:.2e} g (cgs)")

        # 篩選粒子：距離原點至少 min_dist_from_origin (cm)
        if min_dist_from_origin > 0.0:
            distances_to_origin = np.linalg.norm(self.coordinates_cgs, axis=1)
            mask = distances_to_origin >= min_dist_from_origin
            
            if np.sum(mask) == 0:
                print(f"  Warning: No particles found with distance >= {min_dist_from_origin:.2e} cm. All particles filtered out.")
                self.coordinates_cgs = np.array([])
                self.velocities_cgs = np.array([])
                self.masses_cgs = np.array([])
                self.vel_magnitude_cgs = np.array([])
                return

            self.coordinates_cgs = self.coordinates_cgs[mask]
            self.velocities_cgs = self.velocities_cgs[mask]
            self.masses_cgs = self.masses_cgs[mask]
            
            print(f"  Filtered particles: {self.num_particles_read} original, {len(self.coordinates_cgs)} after distance filtering (min_dist={min_dist_from_origin:.2e} cm).")

        # 粒子採樣 (在距離篩選之後進行)
        n_particles = len(self.coordinates_cgs)
        if n_particles > sample_size:
            idx = np.random.choice(n_particles, size=sample_size, replace=False)
            self.coordinates_cgs = self.coordinates_cgs[idx]
            self.velocities_cgs = self.velocities_cgs[idx]
            self.masses_cgs = self.masses_cgs[idx]
            print(f"  Sampled {len(self.coordinates_cgs)} particles from filtered set.")

        # 計算 CGS 單位下的速度幅度
        self.vel_magnitude_cgs = np.sqrt(np.sum(self.velocities_cgs**2, axis=1))

    # --- Getter 方法返回 CGS 單位數據 ---
    def get_coordinates(self):
        """返回 CGS 單位 (cm) 的粒子坐標。"""
        return self.coordinates_cgs

    def get_velocities(self):
        """返回 CGS 單位 (cm/s) 的粒子速度。"""
        return self.velocities_cgs

    def get_masses(self):
        """返回 CGS 單位 (g) 的粒子質量。"""
        return self.masses_cgs

    def get_velocity_magnitude(self):
        """返回 CGS 單位 (cm/s) 的粒子速度幅度。"""
        return self.vel_magnitude_cgs
    
    def get_num_particles_read(self):
        """返回讀取原始快照中的粒子總數。"""
        return self.num_particles_read

    def calculate_kinetic_energy(self):
        """
        計算系統的總動能 (CGS 單位)。
        Returns:
            float: 總動能，單位為 erg (爾格)。
        """
        if self.velocities_cgs is None or self.masses_cgs is None:
            raise ValueError("CGS velocities or masses not loaded. Call process_data() first.")

        total_kinetic_energy = 0.5 * np.sum(self.masses_cgs * self.vel_magnitude_cgs**2)
        return total_kinetic_energy

    def calculate_potential_energy(self, G=6.67e-8, method='direct', max_particles_for_direct=None): 
        """
        計算採樣粒子的引力位能 (CGS 單位)。
        
        Args:
            G (float): 萬有引力常數。單位 cm^3 g^-1 s^-2 (CGS)。
            method (str): 位能計算方法。
                          'direct': O(N^2) 精確計算所有粒子對的位能。
                          'com_approx': 假設質量中心位於 (0,0,0) 的 O(N) 簡化近似。
            max_particles_for_direct (int or None): 使用 'direct' 方法時，允許計算的最大粒子數。
                                                    如果為 None 或大於實際粒子數，則不設限制。
                                                    設置此參數主要是為了避免極度耗時的計算。
        
        Returns:
            float: 採樣粒子的總引力位能，單位為 erg (爾格)。
        """
        if self.coordinates_cgs is None or self.masses_cgs is None:
            raise ValueError("CGS coordinates or masses not loaded. Call process_data() first.")

        if method == 'direct':
            # --- 移除直接返回 0 的硬性限制 ---
            # 現在只發出警告，並繼續嘗試計算，但會非常慢
            if max_particles_for_direct is not None and len(self.masses_cgs) > max_particles_for_direct:
                 print(f"  Warning: Attempting direct O(N^2) potential energy calculation for {len(self.masses_cgs)} particles, which exceeds the recommended limit of {max_particles_for_direct}. This calculation will be EXTREMELY SLOW and may consume significant resources.")
            
            total_potential_energy = self._calculate_direct_potential_energy_numba(
                G, self.coordinates_cgs, self.masses_cgs
            )
            print(f"  Info: Calculated potential energy using direct O(N^2) method (Numba accelerated) for {len(self.masses_cgs)} particles.")
            return total_potential_energy

        elif method == 'com_approx':
            total_mass_of_sampled_system = np.sum(self.masses_cgs)
            print(f"total_mass_of_sampled_system={total_mass_of_sampled_system}")
            distances_to_origin = np.linalg.norm(self.coordinates_cgs, axis=1)
            print(f"distances_to_origin={distances_to_origin}")

            r_epsilon_for_com = 1e-10 # 避免除以零的距離閾值 (cm)
            
            potential_energies_per_particle = -G * total_mass_of_sampled_system * self.masses_cgs / np.maximum(distances_to_origin, r_epsilon_for_com)
            print(f"potential_energies_per_particle={potential_energies_per_particle}")

            total_potential_energy = np.sum(potential_energies_per_particle)
            
            print(f"  Info: Calculated potential energy using simplified O(N) method (assuming COM at origin).")
            return total_potential_energy
        else:
            raise ValueError(f"Unknown potential energy calculation method: {method}. Choose 'direct' or 'com_approx'.")

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def _calculate_direct_potential_energy_numba(G, coordinates, masses):
        """
        Numba 加速的 O(N^2) 直接位能計算。
        """
        num_particles = len(masses)
        total_potential_energy = 0.0
        
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                dx = coordinates[i, 0] - coordinates[j, 0]
                dy = coordinates[i, 1] - coordinates[j, 1]
                dz = coordinates[i, 2] - coordinates[j, 2]
                r_ij_dist_sq = dx**2 + dy**2 + dz**2
                r_ij_dist = np.sqrt(r_ij_dist_sq)
                total_potential_energy += -G * masses[i] * masses[j] / r_ij_dist
        
        return total_potential_energy

    @staticmethod
    def print_structure(file_path):
        def print_attrs(name, obj):
            print(f"\nObject: {name}")
            if isinstance(obj, h5py.Dataset):
                print(f"   Type: Dataset, Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print("   Type: Group")
            if obj.attrs:
                print("   Attributes:")
                for attr_name, attr_value in obj.attrs.items():
                    print(f"     {attr_name}: {attr_value}")

        try:
            with h5py.File(file_path, 'r') as f:
                print(f"\nHDF5 File Structure for: {file_path}")
                f.visititems(print_attrs)
        except Exception as e:
            print(f"Error printing HDF5 structure for {file_path}: {e}")