# -*- encoding: utf-8 -*-
"""
优化版本
"""

import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import os
import logging
from .model import Material

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FieldCache:
    """电场缓存类 """
    
    def __init__(self, my_f, resolution=5.0):  # 增加分辨率适应大型器件
        self.my_f = my_f
        self.resolution = resolution
        self.e_field_cache = {}
        self.doping_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        logger.info(f"电场缓存初始化完成，分辨率: {resolution} um，适用于大型器件")
    
    def get_e_field_cached(self, x, y, z):
        """获取缓存的电场值 """
        try:
            # 更宽松的位置验证
            if not self._is_position_valid(x, y, z):
                return self._safe_get_e_field(x, y, z)
                
            # 使用更粗的分辨率适应大型器件
            key_x = int(round(x / self.resolution))
            key_y = int(round(y / self.resolution))
            key_z = int(round(z / self.resolution))
            key = (key_x, key_y, key_z)
            
            if key in self.e_field_cache:
                self._cache_stats['hits'] += 1
                return self.e_field_cache[key]
            else:
                self._cache_stats['misses'] += 1
                e_field = self._safe_get_e_field(x, y, z)
                if e_field is not None:
                    self.e_field_cache[key] = e_field
                return e_field
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.warning(f"电场缓存获取失败 ({x:.1f}, {y:.1f}, {z:.1f}): {e}")
            return self._safe_get_e_field(x, y, z)
    
    def get_doping_cached(self, x, y, z):
        """获取掺杂浓度"""
        try:
            if not self._is_position_valid(x, y, z):
                return self._safe_get_doping(x, y, z)
                
            key_x = int(round(x / self.resolution))
            key_y = int(round(y / self.resolution))
            key_z = int(round(z / self.resolution))
            key = (key_x, key_y, key_z)
            
            if key in self.doping_cache:
                return self.doping_cache[key]
            else:
                doping = self._safe_get_doping(x, y, z)
                if doping is not None:
                    self.doping_cache[key] = doping
                return doping
        except Exception as e:
            logger.warning(f"掺杂浓度获取失败 ({x:.1f}, {y:.1f}, {z:.1f}): {e}")
            return 0.0  # 默认掺杂浓度
    
    def _is_position_valid(self, x, y, z):
        """位置验证"""
        # 大型器件可能达到 10000 um，设置合理的范围
        max_size = 50000  # 50 mm
        if (abs(x) > max_size or abs(y) > max_size or abs(z) > max_size or
            math.isnan(x) or math.isnan(y) or math.isnan(z) or
            math.isinf(x) or math.isinf(y) or math.isinf(z)):
            return False
        return True
    
    def _safe_get_e_field(self, x, y, z):
        """安全的电场获取"""
        try:
            return self.my_f.get_e_field(x, y, z)
        except Exception as e:
            logger.error(f"电场获取失败，使用默认值: {e}")
            # 返回一个小的默认电场，避免载流子立即停止
            return [0.0, 0.0, 100.0]  # 100 V/cm 的默认电场
    
    def _safe_get_doping(self, x, y, z):
        """安全的掺杂浓度获取"""
        try:
            return self.my_f.get_doping(x, y, z)
        except Exception as e:
            logger.warning(f"掺杂浓度获取失败，使用默认值: {e}")
            return 0.0
    
    def get_cache_stats(self):
        """获取缓存统计"""
        total = self._cache_stats['hits'] + self._cache_stats['misses'] + self._cache_stats['errors']
        hit_rate = self._cache_stats['hits'] / total if total > 0 else 0
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'errors': self._cache_stats['errors'],
            'hit_rate': hit_rate,
            'total_entries': len(self.e_field_cache)
        }

class VectorizedCarrierSystem:
    """向量化载流子系统 """
    
    def __init__(self, all_positions, all_charges, all_times, material, carrier_type="electron", 
                 read_out_contact=None, my_d=None):
        # 输入数据验证
        self._validate_inputs(all_positions, all_charges, all_times)
            
        # 初始化数组
        self.positions = np.array(all_positions, dtype=np.float64)
        self.charges = np.array(all_charges, dtype=np.float64)
        self.times = np.array(all_times, dtype=np.float64)
        self.active = np.ones(len(all_charges), dtype=bool)
        self.end_conditions = np.zeros(len(all_charges), dtype=np.int8)
        self.steps_drifted = np.zeros(len(all_charges), dtype=np.int32)
        self.carrier_type = carrier_type
        self.read_out_contact = read_out_contact
        self.my_d = my_d
        
        # Material 对象创建
        self.material = self._create_material_safe(material)
        
        # 探测器参数提取
        self.detector_params = self._extract_detector_params_robust(my_d)
        
        # 初始化其他属性
        self._initialize_other_attributes(all_positions)
        
        # 物理常数
        self.kboltz = 8.617385e-5
        self.e0 = 1.60217733e-19
        
        # 性能统计
        self.performance_stats = {
            'total_steps': 0,
            'field_calculations': 0,
            'boundary_checks': 0,
            'carriers_terminated': 0,
            'low_field_terminations': 0,
            'boundary_terminations': 0
        }
        
        logger.info(f"向量化系统初始化: {len(all_charges)}个{carrier_type}")
        logger.info(f"探测器尺寸: {self.detector_params['l_x']:.1f} × {self.detector_params['l_y']:.1f} × {self.detector_params['l_z']:.1f} um")
    
    def _validate_inputs(self, positions, charges, times):
        """输入数据验证"""
        if len(positions) == 0:
            raise ValueError("载流子位置列表不能为空")
        if len(positions) != len(charges) or len(positions) != len(times):
            raise ValueError("位置、电荷和时间数组长度不一致")
        
        # 检查位置数据有效性
        for i, pos in enumerate(positions):
            if len(pos) != 3:
                raise ValueError(f"位置数据 {i} 格式错误，应为 [x, y, z]")
            x, y, z = pos
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                raise ValueError(f"位置数据 {i} 包含 NaN 值")
    
    def _create_material_safe(self, material):
        """安全的 Material 对象创建"""
        try:
            return Material(material)
        except Exception as e:
            logger.warning(f"Material对象创建失败 {material}，使用默认硅材料: {e}")
            try:
                return Material("si")
            except:
                # 最终备用方案
                class FallbackMaterial:
                    def __init__(self):
                        self.name = "fallback_si"
                return FallbackMaterial()
    
    def _extract_detector_params_robust(self, my_d):
        """探测器参数提取 """
        params = {}
        try:
            if my_d is not None:
                # 核心尺寸参数
                params['l_x'] = self._get_param_safe(my_d, 'l_x', 10000.0)
                params['l_y'] = self._get_param_safe(my_d, 'l_y', 10000.0) 
                params['l_z'] = self._get_param_safe(my_d, 'l_z', 300.0)
                
                # 像素参数
                params['p_x'] = self._get_param_safe(my_d, 'p_x', 50.0)
                params['p_y'] = self._get_param_safe(my_d, 'p_y', 50.0)
                
                # 电极数量
                params['n_x'] = self._get_param_safe(my_d, 'x_ele_num', 200, param_type=int)
                params['n_y'] = self._get_param_safe(my_d, 'y_ele_num', 200, param_type=int)
                
                # 其他参数
                params['field_shift_x'] = self._get_param_safe(my_d, 'field_shift_x', 0.0)
                params['field_shift_y'] = self._get_param_safe(my_d, 'field_shift_y', 0.0)
                params['temperature'] = self._get_param_safe(my_d, 'temperature', 300.0)
                
                # 大型器件专用参数
                params['boundary_tolerance'] = 1.0  # 增加边界容差
                params['max_drift_time'] = 100e-9   # 增加最大漂移时间
                params['min_field_strength'] = 1.0  # 降低电场阈值
                
                logger.info("探测器参数提取成功")
                
            else:
                # 大型器件合理的默认值
                params.update(self._get_large_detector_defaults())
                logger.warning("my_d 为 None，使用大型器件默认参数")
                
        except Exception as e:
            logger.error(f"探测器参数提取失败: {e}")
            params.update(self._get_large_detector_defaults())
            
        return params
    
    def _get_param_safe(self, my_d, param_name, default, param_type=float):
        """安全获取参数"""
        try:
            value = getattr(my_d, param_name, default)
            return param_type(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"参数 {param_name} 转换失败，使用默认值 {default}: {e}")
            return default
    
    def _get_large_detector_defaults(self):
        """大型器件默认参数"""
        return {
            'l_x': 10000.0, 'l_y': 10000.0, 'l_z': 300.0,
            'p_x': 50.0, 'p_y': 50.0,
            'n_x': 200, 'n_y': 200,
            'field_shift_x': 0.0, 'field_shift_y': 0.0,
            'temperature': 300.0,
            'boundary_tolerance': 1.0,
            'max_drift_time': 100e-9,
            'min_field_strength': 1.0
        }
    
    def _initialize_other_attributes(self, all_positions):
        """初始化其他属性"""
        # 初始化 reduced_positions
        self.reduced_positions = np.zeros((len(all_positions), 2), dtype=np.float64)
        for i, pos in enumerate(all_positions):
            x, y, z = pos
            x_reduced, y_reduced = self._calculate_reduced_coords(x, y, self.my_d)
            self.reduced_positions[i] = [x_reduced, y_reduced]
        
        # 存储路径
        self.paths = [[] for _ in range(len(all_positions))]
        self.paths_reduced = [[] for _ in range(len(all_positions))]
        
        # 初始化路径数据
        for i in range(len(all_positions)):
            x, y, z = all_positions[i]
            t = self.times[i]
            self.paths[i].append([x, y, z, t])
            
            x_reduced, y_reduced = self.reduced_positions[i]
            x_num, y_num = self._calculate_electrode_numbers(x, y)
            self.paths_reduced[i].append([x_reduced, y_reduced, z, t, x_num, y_num])
    
    def _calculate_reduced_coords(self, x, y, my_d):
        """计算简化坐标"""
        params = self.detector_params
        
        use_reduced = (self.read_out_contact and 
                      len(self.read_out_contact) == 1 and
                      (self.read_out_contact[0].get('x_span', 0) != 0 or 
                       self.read_out_contact[0].get('y_span', 0) != 0))
        
        if use_reduced:
            x_reduced = (x - params['l_x']/2) % params['p_x'] + params['field_shift_x']
            y_reduced = (y - params['l_y']/2) % params['p_y'] + params['field_shift_y']
        else:
            x_reduced = x
            y_reduced = y
        
        return x_reduced, y_reduced
    
    def _calculate_electrode_numbers(self, x, y):
        """计算电极编号"""
        params = self.detector_params
        try:
            x_num = int((x - params['l_x']/2) // params['p_x'] + params['n_x']/2.0)
            y_num = int((y - params['l_y']/2) // params['p_y'] + params['n_y']/2.0)
            # 确保电极编号在合理范围内
            x_num = max(0, min(params['n_x']-1, x_num))
            y_num = max(0, min(params['n_y']-1, y_num))
            return x_num, y_num
        except Exception as e:
            # 返回中心电极
            return params['n_x']//2, params['n_y']//2

    def _calculate_correct_mobility(self, temperature, doping, charge, electric_field):
        """迁移率计算 """
        try:
            field_strength = np.linalg.norm(electric_field)
            
            # 硅的基本迁移率
            if charge > 0:  # 空穴
                mu_low_field = 480.0
                beta = 1.0
                vsat = 0.95e7
            else:  # 电子
                mu_low_field = 1350.0
                beta = 2.0
                vsat = 1.0e7
            
            # 高电场速度饱和模型
            if field_strength > 1e3:
                E0 = vsat / mu_low_field
                mu = mu_low_field / (1 + (field_strength / E0) ** beta) ** (1 / beta)
                mu = max(mu, vsat / field_strength)
            else:
                mu = mu_low_field
            
            return mu
        except Exception as e:
            logger.warning(f"迁移率计算失败，使用默认值: {e}")
            return 1350.0 if charge < 0 else 480.0

    def _check_boundary_conditions(self, x, y, z):
        """边界条件检查 """
        params = self.detector_params
        l_x, l_y, l_z = params['l_x'], params['l_y'], params['l_z']
        tolerance = params['boundary_tolerance']
        
        # 使用容差检查边界
        out_of_bound = (x <= -tolerance or x >= l_x + tolerance or 
                       y <= -tolerance or y >= l_y + tolerance or 
                       z <= -tolerance or z >= l_z + tolerance)
        
        return out_of_bound

    def drift_batch(self, my_d, field_cache, delta_t=1e-12, max_steps=5000):
        """批量漂移主函数 """
        logger.info(f"开始批量漂移{self.carrier_type}，最多{max_steps}步，时间步长{delta_t}s")
        
        start_time = time.time()
        delta_t_cm = delta_t * 1e4
        
        total_carriers = len(self.active)
        initial_active = np.sum(self.active)
        
        logger.info(f"初始状态: {initial_active}/{total_carriers} 个活跃载流子")
        
        for step in range(max_steps):
            if step % 100 == 0:
                self._log_progress(step, total_carriers)
            
            n_terminated = self.drift_step_batch(my_d, field_cache, delta_t, delta_t_cm, step)
            self.performance_stats['total_steps'] += 1
            
            if not np.any(self.active):
                logger.info("所有载流子停止漂移")
                break
        
        self._log_final_stats(start_time, max_steps)
        return True

    def drift_step_batch(self, my_d, field_cache, delta_t, delta_t_cm, step=0):
        """批量单步漂移 """
        if not np.any(self.active):
            return 0
            
        n_terminated = 0
        params = self.detector_params
        
        # 预计算扩散常数
        diffusion_constant = math.sqrt(2.0 * self.kboltz * params['temperature'] * delta_t) * 1e4
        
        for idx in range(len(self.active)):
            if not self.active[idx]:
                continue
                
            x, y, z = self.positions[idx]
            charge = self.charges[idx]
            
            # 边界检查
            self.performance_stats['boundary_checks'] += 1
            if self._check_boundary_conditions(x, y, z):
                self.active[idx] = False
                self.end_conditions[idx] = 1
                n_terminated += 1
                self.performance_stats['boundary_terminations'] += 1
                continue
            
            # 时间检查
            if self.times[idx] > params['max_drift_time']:
                self.active[idx] = False
                self.end_conditions[idx] = 4
                n_terminated += 1
                continue
            
            # 电场获取和处理
            e_field = self._get_e_field_safe(field_cache, x, y, z, idx)
            if e_field is None:
                continue
                
            Ex, Ey, Ez = e_field
            intensity = math.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)
            
            # 电场强度检查（降低阈值）
            if intensity <= params['min_field_strength']:
                self.active[idx] = False
                self.end_conditions[idx] = 3
                n_terminated += 1
                self.performance_stats['low_field_terminations'] += 1
                continue
            
            # 迁移率计算
            try:
                doping = field_cache.get_doping_cached(x, y, z)
                mu = self._calculate_correct_mobility(params['temperature'], doping, charge, e_field)
            except Exception as e:
                mu = 1350.0 if charge < 0 else 480.0
            
            # 速度和位移计算
            delta_x, delta_y, delta_z = self._calculate_displacement(charge, e_field, mu, delta_t_cm)
            
            # 扩散位移
            dif_x, dif_y, dif_z = self._calculate_diffusion(diffusion_constant, mu)
            
            # 更新位置
            self._update_carrier_position(idx, delta_x, delta_y, delta_z, dif_x, dif_y, dif_z, delta_t)
        
        self.performance_stats['carriers_terminated'] += n_terminated
        return n_terminated

    def _get_e_field_safe(self, field_cache, x, y, z, idx):
        """安全的电场获取"""
        try:
            self.performance_stats['field_calculations'] += 1
            e_field = field_cache.get_e_field_cached(x, y, z)
            if e_field is None or len(e_field) != 3:
                raise ValueError("无效的电场值")
            return e_field
        except Exception as e:
            logger.warning(f"载流子 {idx} 电场获取失败: {e}")
            self.active[idx] = False
            self.end_conditions[idx] = 2
            return None

    def _calculate_displacement(self, charge, e_field, mu, delta_t_cm):
        """计算位移"""
        Ex, Ey, Ez = e_field
        if charge > 0:  # 空穴
            vx = Ex * mu
            vy = Ey * mu
            vz = Ez * mu
        else:  # 电子
            vx = -Ex * mu
            vy = -Ey * mu
            vz = -Ez * mu
        
        return vx * delta_t_cm, vy * delta_t_cm, vz * delta_t_cm

    def _calculate_diffusion(self, diffusion_constant, mu):
        """计算扩散位移"""
        try:
            diffusion = diffusion_constant * math.sqrt(mu)
            return (random.gauss(0.0, diffusion),
                   random.gauss(0.0, diffusion), 
                   random.gauss(0.0, diffusion))
        except:
            return 0.0, 0.0, 0.0

    def _update_carrier_position(self, idx, delta_x, delta_y, delta_z, dif_x, dif_y, dif_z, delta_t):
        """更新载流子位置"""
        x, y, z = self.positions[idx]
        
        new_x = x + delta_x + dif_x
        new_y = y + delta_y + dif_y
        new_z = z + delta_z + dif_z
        
        # 更新坐标
        self.positions[idx] = [new_x, new_y, new_z]
        self.reduced_positions[idx] = self._calculate_reduced_coords(new_x, new_y, self.my_d)
        self.times[idx] += delta_t
        self.steps_drifted[idx] += 1
        
        # 更新路径
        self.paths[idx].append([new_x, new_y, new_z, self.times[idx]])
        x_num, y_num = self._calculate_electrode_numbers(new_x, new_y)
        self.paths_reduced[idx].append([
            self.reduced_positions[idx][0], self.reduced_positions[idx][1], 
            new_z, self.times[idx], x_num, y_num
        ])

    def _log_progress(self, step, total_carriers):
        """记录进度"""
        active_count = np.sum(self.active)
        progress = (total_carriers - active_count) / total_carriers * 100
        logger.info(f"  步骤 {step}: {active_count}个活跃载流子 ({progress:.1f}%完成)")

    def _log_final_stats(self, start_time, max_steps):
        """记录最终统计"""
        end_time = time.time()
        total_time = end_time - start_time
        final_stats = self.get_statistics()
        perf_stats = self.get_performance_stats()
        
        logger.info(f"批量漂移完成: 共{self.performance_stats['total_steps']}步，耗时{total_time:.2f}秒")
        logger.info(f"最终状态: {final_stats['active_carriers']}个活跃，平均步数{final_stats['average_steps']:.1f}")
        logger.info(f"性能统计: {perf_stats}")

    def get_statistics(self):
        """获取统计信息"""
        n_total = len(self.active)
        n_active = np.sum(self.active)
        
        if np.any(self.steps_drifted > 0):
            avg_steps = np.mean(self.steps_drifted[self.steps_drifted > 0])
            max_steps = np.max(self.steps_drifted)
        else:
            avg_steps = 0
            max_steps = 0
            
        # 终止原因统计
        end_condition_counts = {
            'boundary': np.sum(self.end_conditions == 1),
            'field_error': np.sum(self.end_conditions == 2),
            'low_field': np.sum(self.end_conditions == 3),
            'timeout': np.sum(self.end_conditions == 4),
            'active': n_active
        }
        
        return {
            'total_carriers': n_total,
            'active_carriers': n_active,
            'inactive_carriers': n_total - n_active,
            'average_steps': avg_steps,
            'max_steps': max_steps,
            'carrier_type': self.carrier_type,
            'end_conditions': end_condition_counts
        }
    
    def get_performance_stats(self):
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def update_original_carriers(self, original_carriers):
        """更新原始载流子对象"""
        logger.info(f"更新{self.carrier_type}状态...")
        updated_count = 0
        
        for i, carrier in enumerate(original_carriers):
            if i < len(self.positions):
                try:
                    # 更新基本属性
                    carrier.x = float(self.positions[i][0])
                    carrier.y = float(self.positions[i][1])
                    carrier.z = float(self.positions[i][2])
                    carrier.t = float(self.times[i])
                    
                    # 更新其他属性
                    x_reduced, y_reduced = self.reduced_positions[i]
                    carrier.x_reduced = float(x_reduced)
                    carrier.y_reduced = float(y_reduced)
                    
                    x_num, y_num = self._calculate_electrode_numbers(carrier.x, carrier.y)
                    carrier.x_num = x_num
                    carrier.y_num = y_num
                    
                    # 更新路径
                    carrier.path = [[float(p[0]), float(p[1]), float(p[2]), float(p[3])] 
                                  for p in self.paths[i]]
                    
                    carrier.path_reduced = [[
                        float(p[0]), float(p[1]), float(p[2]), 
                        float(p[3]), int(p[4]), int(p[5])
                    ] for p in self.paths_reduced[i]]
                    
                    # 重新初始化信号列表
                    self._reinitialize_signal_list(carrier)
                    
                    # 更新终止条件
                    if not self.active[i]:
                        condition_map = {1: "超出边界", 2: "电场错误", 3: "低电场", 4: "超时"}
                        carrier.end_condition = condition_map.get(self.end_conditions[i], "未知")
                    
                    updated_count += 1
                    
                except Exception as e:
                    logger.warning(f"更新载流子 {i} 时出错: {e}")
        
        logger.info(f"已更新 {updated_count} 个{self.carrier_type}")
        return updated_count
    
    def _reinitialize_signal_list(self, carrier):
        """重新初始化信号列表"""
        try:
            if hasattr(carrier, 'read_out_contact') and carrier.read_out_contact:
                if len(carrier.read_out_contact) == 1:
                    x_span = carrier.read_out_contact[0].get('x_span', 0)
                    y_span = carrier.read_out_contact[0].get('y_span', 0)
                    carrier.signal = [[] for _ in range((2*x_span+1)*(2*y_span+1))]
                else:
                    carrier.signal = [[] for _ in range(len(carrier.read_out_contact))]
            else:
                carrier.signal = [[]]
        except Exception as e:
            carrier.signal = [[]]

# 在 cal_current.py 中调用的函数
def generate_electron_images(electron_system, save_dir="electron_images"):
    """生成电子图像的主函数"""
    if electron_system.carrier_type == "electron":
        electron_system.generate_electron_images(save_dir)
        logger.info("电子图像生成完成！")
    else:
        logger.error("错误：提供的载流子系统不是电子类型")
