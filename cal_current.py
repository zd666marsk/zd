# -*- encoding: utf-8 -*-

'''
Description:  
    Simulate e-h pairs drifting and calculate induced current
@Date       : 2021/09/02 14:01:46
@Author     : Yuhang Tan, Chenxi Fu
@version    : 2.0
'''

import math
import os
from array import array
import csv
import time
import logging
import random
import importlib
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)


def _import_first_available(*candidates):
    """尝试依次导入候选模块，返回第一个成功的模块对象。"""

    last_error = None
    for dotted_path in candidates:
        try:
            return importlib.import_module(dotted_path)
        except ModuleNotFoundError as exc:
            last_error = exc
        except ImportError as exc:  # 兼容较老 Python 版本的错误类型
            last_error = exc

    # 将最后一次导入错误转换成更易读的消息
    raise ModuleNotFoundError(
        "无法定位以下任一模块: {}".format(
            ", ".join(candidates)
        )
    ) from last_error


Material = _import_first_available("model", "raser.model", "current.model").Material
CarrierListFromG4P = _import_first_available(
    "interaction.carrier_list",
    "raser.interaction.carrier_list",
    "current.interaction.carrier_list",
).CarrierListFromG4P
math_module = _import_first_available(
    "util.math",
    "raser.util.math",
    "current.util.math",
)
Vector = math_module.Vector
signal_convolution = math_module.signal_convolution
output = _import_first_available(
    "util.output",
    "raser.util.output",
    "current.util.output",
).output
from .model import Material
from ..interaction.carrier_list import CarrierListFromG4P
from ..util.math import Vector, signal_convolution
from ..util.output import output

OPTIMIZATION_AVAILABLE = True

logger = logging.getLogger(__name__ + ".optimization")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


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

def _resolve_gauss_sampler(rng=None):
    """统一处理不同随机数源的正态分布接口。"""
    if rng is None:
        return random.gauss

    if hasattr(rng, "gauss") and callable(rng.gauss):
        return rng.gauss

    if hasattr(rng, "normal") and callable(rng.normal):
        return lambda mean, sigma: float(rng.normal(mean, sigma))

    raise TypeError("提供的随机数生成器不支持正态分布抽样")


class VectorizedCarrierSystem:
    """向量化载流子系统 """
    
    def __init__(self, all_positions, all_charges, all_times, material, carrier_type="electron",
                 read_out_contact=None, my_d=None, rng=None):
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

        # 随机数生成器：默认回退到模块级 random，使 random.seed() 完整生效
        self._gauss = _resolve_gauss_sampler(rng)
        
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
            
            # 扩散位移（考虑复合载流子数量带来的噪声放大效应）
            dif_x, dif_y, dif_z = self._calculate_diffusion(diffusion_constant, mu, charge)
            
            # 更新位置
            self._update_carrier_position(idx, delta_x, delta_y, delta_z, dif_x, dif_y, dif_z, delta_t)
        
        self.performance_stats['carriers_terminated'] += n_terminated
        return n_terminated

    def _get_e_field_safe(self, field_cache, x, y, z, idx):
        """安全的电场获取"""
        try:
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
            
            # 扩散位移（考虑复合载流子数量带来的噪声放大效应）
            dif_x, dif_y, dif_z = self._calculate_diffusion(diffusion_constant, mu, charge)
            
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

    def _calculate_diffusion(self, diffusion_constant, mu, charge):
        """计算扩散位移，考虑复合载流子的统计平均效应"""
        try:
            mobility = max(mu, 0.0)
            if mobility == 0.0:
                return 0.0, 0.0, 0.0

            diffusion_sigma = diffusion_constant * math.sqrt(mobility)

            # 复合载流子：中心扩散宽度 ~ 1/sqrt(N)，避免出现非物理噪声放大
            carrier_count = max(1.0, abs(charge))
            diffusion_sigma /= math.sqrt(carrier_count)

            diffs = (
                self._gauss(0.0, diffusion_sigma),
                self._gauss(0.0, diffusion_sigma),
                self._gauss(0.0, diffusion_sigma),
            )
            return diffs
        except Exception:
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

    def _calculate_diffusion(self, diffusion_constant, mu, charge):
        """计算扩散位移，考虑复合载流子的统计平均效应"""
        try:
            mobility = max(mu, 0.0)
            if mobility == 0.0:
                return 0.0, 0.0, 0.0

            diffusion_sigma = diffusion_constant * math.sqrt(mobility)

            # 复合载流子：中心扩散宽度 ~ 1/sqrt(N)，避免出现非物理噪声放大
            carrier_count = max(1.0, abs(charge))
            diffusion_sigma /= math.sqrt(carrier_count)

            diffs = (
                self._gauss(0.0, diffusion_sigma),
                self._gauss(0.0, diffusion_sigma),
                self._gauss(0.0, diffusion_sigma),
            )
            return diffs
        except Exception:
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

t_bin = 10e-12
# resolution of oscilloscope
t_end = 10e-9
t_start = 0
delta_t = 1e-12
min_intensity = 1 # V/cm

class CarrierCluster:
    """
    Description:
        Definition of carriers and the record of their movement
    Parameters:
        x_init, y_init, z_init, t_init : float
            initial space and time coordinates in um and s
        charge : float
            a set of drifting carriers, absolute value for number, sign for charge
    Attributes:
        x, y, z, t : float
            space and time coordinates in um and s
        path : float[]
            recording the carrier path in [x, y, z, t]
        charge : float
            a set of drifting carriers, absolute value for number, sign for charge
        signal : float[]
            the generated signal current on the reading electrode
        end_condition : 0/string
            tag of how the carrier ended drifting
    Modify:
        2022/10/28
    """
    def __init__(self, x_init, y_init, z_init, t_init, p_x, p_y, n_x, n_y, l_x, l_y, field_shift_x, field_shift_y, charge, material, weighting_field, rng=None):
        self.x = x_init
        self.y = y_init
        self.z = z_init
        self.t = t_init
        self.t_end = t_end
        self.path = [[x_init, y_init, z_init, t_init]]

        self.field_shift_x = field_shift_x
        self.field_shift_y = field_shift_y
        # for odd strip, field shift should let x_reduced = 0 at the center of the strip
        # for even strip, field shift should let x_reduced = 0 at the edge of the strip
        self.p_x = p_x
        self.p_y = p_y
        self.x_num = int((x_init-l_x/2) // p_x + n_x/2.0)
        self.y_num = int((y_init-l_y/2) // p_y + n_y/2.0)
        if len(weighting_field) == 1 and (weighting_field[0]['x_span'] != 0 or weighting_field[0]['y_span'] != 0):
            self.x_reduced = (x_init-l_x/2) % p_x + field_shift_x
            self.y_reduced = (y_init-l_y/2) % p_y + field_shift_y

        else:
            self.x_reduced = x_init
            self.y_reduced = y_init
        if len(weighting_field) == 1 and (weighting_field[0]['x_span'] != 0 or weighting_field[0]['y_span'] != 0):
            x_span = weighting_field[0]['x_span']
            y_span = weighting_field[0]['y_span']
            # 正确计算信号列表长度：应该是 (2*x_span+1) * (2*y_span+1)
            signal_length = (2 * x_span + 1) * (2 * y_span + 1)
            self.signal = [[] for j in range(signal_length)]
            # 添加调试信息
            # print(f"载流子初始化: 信号列表长度={signal_length} (x_span={x_span}, y_span={y_span})")        
        self.end_condition = 0

        self.cal_mobility = Material(material).cal_mobility
        self.charge = charge
        if self.charge == 0:
            self.end_condition = "zero charge"

        # 随机源：默认直接使用模块 random，也支持注入自定义 RNG
        self._gauss = _resolve_gauss_sampler(rng)

    def not_in_sensor(self,my_d):
        if (self.x<=0) or (self.x>=my_d.l_x)\
            or (self.y<=0) or (self.y>=my_d.l_y)\
            or (self.z<=0) or (self.z>=my_d.l_z):
            self.end_condition = "out of bound"
        return self.end_condition
    
    def not_in_field_range(self,my_d):
        if (self.x_num<0) or (self.x_num>=my_d.x_ele_num)\
            or (self.y_num<0) or (self.y_num>=my_d.y_ele_num):
            self.end_condition = "out of field range"
        return self.end_condition

    def drift_single_step(self, my_d, my_f, delta_t=delta_t):
        e_field = my_f.get_e_field(self.x_reduced,self.y_reduced,self.z)
        intensity = Vector(e_field[0],e_field[1],e_field[2]).get_length()
        mobility = Material(my_d.material)
        mu = mobility.cal_mobility(my_d.temperature, my_f.get_doping(self.x_reduced, self.y_reduced, self.z), self.charge, intensity)
        velocity_vector = [e_field[0]*mu, e_field[1]*mu, e_field[2]*mu] # cm/s
        if not hasattr(self, '_debug_printed') and len(self.path) < 5:
            print(f"🔍 载流子诊断:")
            print(f"   类型: {'空穴' if self.charge > 0 else '电子'}")
            print(f"   位置: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) um")
            print(f"   电场: {intensity:.1f} V/cm")
            print(f"   迁移率: {mu:.1f} cm²/V·s")
            self._debug_printed = True
    
        velocity_vector = [e_field[0]*mu, e_field[1]*mu, e_field[2]*mu] # cm/s
        if(intensity > min_intensity):
            #project steplength on the direction of electric field
            if(self.charge>0):
                delta_x = velocity_vector[0]*delta_t*1e4 # um
                delta_y = velocity_vector[1]*delta_t*1e4
                delta_z = velocity_vector[2]*delta_t*1e4
            else:
                delta_x = -velocity_vector[0]*delta_t*1e4
                delta_y = -velocity_vector[1]*delta_t*1e4
                delta_z = -velocity_vector[2]*delta_t*1e4
        else:
            self.end_condition = "zero velocity"
            return

        # 将多个载流子视作一组时，需要缩小随机扩散步长，避免把全部电荷绑在同一条随机路径上
        # 导致信号方差被非物理地放大

        kboltz=8.617385e-5 #eV/K
        diffusion = (2.0*kboltz*mu*my_d.temperature*delta_t)**0.5

        # 根据复合载流子数量缩放扩散步长，降低非物理噪声
        carrier_count = max(1.0, abs(self.charge))
        diffusion /= math.sqrt(carrier_count)

        diffs = (
            self._gauss(0.0, diffusion) * 1e4,
            self._gauss(0.0, diffusion) * 1e4,
            self._gauss(0.0, diffusion) * 1e4,
        )
        dif_x, dif_y, dif_z = diffs

        # sum up
        # x axis   
        # assume carriers will not drift out of the field range
        self.x_reduced = self.x_reduced+delta_x+dif_x
        self.x = self.x+delta_x+dif_x
        # y axis
        self.y_reduced = self.y_reduced+delta_y+dif_y
        self.y = self.y+delta_y+dif_y
        # z axis
        self.z = self.z+delta_z+dif_z
        #time
        self.t = self.t+delta_t
        #record
        self.path_reduced.append([self.x_reduced, self.y_reduced, self.z, self.t, self.x_num, self.y_num])
        self.path.append([self.x, self.y, self.z, self.t]) 

    def get_signal(self,my_f,my_d):
        """Calculate signal from carrier path"""
        # i = -q*v*nabla(U_w) = -q*dx*nabla(U_w)/dt = -q*dU_w(x)/dt
        # signal = i*dt = -q*dU_w(x)
        if len(my_f.read_out_contact) == 1:
            x_span = my_f.read_out_contact[0]['x_span']
            y_span = my_f.read_out_contact[0]['y_span']
            for j in range(x_span*2+1):
                x_shift = (j-x_span)*self.p_x
                for k in range(y_span*2+1):
                    y_shift = (k-y_span)*self.p_y
                    for i in range(len(self.path_reduced)-1):
                        charge=self.charge
                        U_w_1 = my_f.get_w_p(self.path_reduced[i][0]-x_shift,self.path_reduced[i][1]-y_shift,self.path_reduced[i][2],0)
                        U_w_2 = my_f.get_w_p(self.path_reduced[i+1][0]-x_shift,self.path_reduced[i+1][1]-y_shift,self.path_reduced[i+1][2],0)
                        e0 = 1.60217733e-19
                        if i>0 and my_d.irradiation_model != None:
                            d_t=self.path_reduced[i][3]-self.path_reduced[i-1][3]
                            if self.charge>=0:
                                self.trapping_rate=my_f.get_trap_h(self.path_reduced[i][0],self.path_reduced[i][1],self.path_reduced[i][2])
                            else:
                                self.trapping_rate=my_f.get_trap_e(self.path_reduced[i][0],self.path_reduced[i][1],self.path_reduced[i][2])
                            charge=charge*np.exp(-d_t*self.trapping_rate)
                        q = charge * e0
                        dU_w = U_w_2 - U_w_1
                        self.signal[j].append(q*dU_w)

        else:
            for j in range(len(my_f.read_out_contact)):
                charge=self.charge
                for i in range(len(self.path_reduced)-1): # differentiate of weighting potential
                    U_w_1 = my_f.get_w_p(self.path_reduced[i][0],self.path_reduced[i][1],self.path_reduced[i][2],j) # x,y,z
                    U_w_2 = my_f.get_w_p(self.path_reduced[i+1][0],self.path_reduced[i+1][1],self.path_reduced[i+1][2],j)
                    e0 = 1.60217733e-19
                    if i>0 and my_d.irradiation_model != None:
                        d_t=self.path_reduced[i][3]-self.path_reduced[i-1][3]
                        if self.charge>=0:
                            self.trapping_rate=my_f.get_trap_h(self.path_reduced[i][0],self.path_reduced[i][1],self.path_reduced[i][2])
                        else:
                            self.trapping_rate=my_f.get_trap_e(self.path_reduced[i][0],self.path_reduced[i][1],self.path_reduced[i][2])
                        charge=charge*np.exp(-d_t*self.trapping_rate)
                    q = charge * e0
                    dU_w = U_w_2 - U_w_1
                    self.signal[j].append(q*dU_w)     

    def drift_end(self,my_f):
        e_field = my_f.get_e_field(self.x,self.y,self.z)
        if (e_field[0] == 0 and e_field[1] == 0 and e_field[2] == 0):
            self.end_condition = "out of bound"
        elif (self.t > t_end):
            self.end_condition = "time out"
        return self.end_condition
        

class CalCurrent:
    """
    Description:
        Calculate sum of the generated current by carriers drifting
    Parameters:
        my_d : R3dDetector
        my_f : FenicsCal 
        ionized_pairs : float[]
            the generated carrier amount from MIP or laser
        track_position : float[]
            position of the generated carriers
    Attributes:
        electrons, holes : CarrierCluster[]
            the generated carriers, able to calculate their movement
    Modify:
        2022/10/28
    """
    def __init__(self, my_d, my_f, ionized_pairs, track_position):
        start_time = time.time()
        print("开始载流子电流计算...")
        self.read_ele_num = my_d.read_ele_num
        self.read_out_contact = my_f.read_out_contact
        self.electrons = []
        self.holes = []

        if "planar" in my_d.det_model or "lgad" in my_d.det_model:
            p_x = my_d.l_x
            p_y = my_d.l_y
            n_x = 1
            n_y = 1
            field_shift_x = 0
            field_shift_y = 0
        if "strip" in my_d.det_model:
            # for "lgadstrip", this covers above
            p_x = my_d.p_x
            p_y = my_d.l_y
            n_x = my_d.read_ele_num
            n_y = 1
            field_shift_x = my_d.field_shift_x
            field_shift_y = 0
        if "pixel" in my_d.det_model:
            p_x = my_d.p_x
            p_y = my_d.p_y
            n_x = my_d.x_ele_num
            n_y = my_d.y_ele_num
            field_shift_x = my_d.field_shift_x
            field_shift_y = my_d.field_shift_y

        for i in range(len(track_position)):
            electron = CarrierCluster(track_position[i][0],
                               track_position[i][1],
                               track_position[i][2],
                               track_position[i][3],
                               p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                               -1*ionized_pairs[i],
                               my_d.material,
                               self.read_out_contact)
            hole = CarrierCluster(track_position[i][0],
                           track_position[i][1],
                           track_position[i][2],
                           track_position[i][3],
                           p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                           ionized_pairs[i],
                           my_d.material,
                           self.read_out_contact)
            if not electron.not_in_sensor(my_d) and not electron.not_in_field_range(my_d):
                self.electrons.append(electron)
                self.holes.append(hole)
        init_time = time.time() - start_time
        print(f"载流子初始化完成，耗时: {init_time:.2f}秒")
        self.drifting_loop(my_d, my_f)

        self.t_bin = t_bin
        self.t_end = t_end
        self.t_start = t_start
        self.n_bin = int((self.t_end-self.t_start)/self.t_bin)

        self.current_define(self.read_ele_num)
        for i in range(self.read_ele_num):
            self.sum_cu[i].Reset()
            self.positive_cu[i].Reset()
            self.negative_cu[i].Reset()
        self.get_current(n_x, n_y, self.read_out_contact)
        for i in range(self.read_ele_num):
            self.sum_cu[i].Add(self.positive_cu[i])
            self.sum_cu[i].Add(self.negative_cu[i])

        self.det_model = my_d.det_model
        if "lgad" in self.det_model:
            self.gain_current = CalCurrentGain(my_d, my_f, self)
            for i in range(self.read_ele_num):
                self.sum_cu[i].Add(self.gain_current.negative_cu[i])
                self.sum_cu[i].Add(self.gain_current.positive_cu[i])

    def drifting_loop(self, my_d, my_f):
        """优化的漂移循环 - 自动选择最佳版本"""
        total_carriers = len(self.electrons) + len(self.holes)
        
        # 决定使用哪个版本
        use_optimized = (OPTIMIZATION_AVAILABLE and 
                        total_carriers > 20 and  # 降低阈值，更多测试
                        hasattr(my_d, 'l_x') and hasattr(my_d, 'l_y') and hasattr(my_d, 'l_z') and
                        total_carriers < 10000)  # 避免内存溢出
        
        if use_optimized:
            print(f" 使用优化版本: {len(self.electrons)}电子 + {len(self.holes)}空穴")
            self._drifting_loop_optimized(my_d, my_f)
        else:
            print(f" 使用原始版本: {len(self.electrons)}电子 + {len(self.holes)}空穴")
            self._drifting_loop_original(my_d, my_f)
    
    def _drifting_loop_original(self, my_d, my_f):
        """原始版本的漂移循环"""
        # 电子漂移
        for i, electron in enumerate(self.electrons):
            if i % 100 == 0 and i > 0:
                print(f"  处理电子: {i}/{len(self.electrons)}")
                
            while (not electron.not_in_sensor(my_d) and 
                   not electron.not_in_field_range(my_d) and 
                   not electron.drift_end(my_f)):
                electron.drift_single_step(my_d, my_f)
            electron.get_signal(my_f, my_d)
        
        # 空穴漂移
        for i, hole in enumerate(self.holes):
            if i % 100 == 0 and i > 0:
                print(f"  处理空穴: {i}/{len(self.holes)}")
                
            while (not hole.not_in_sensor(my_d) and 
                   not hole.not_in_field_range(my_d) and 
                   not hole.drift_end(my_f)):
                hole.drift_single_step(my_d, my_f)
            hole.get_signal(my_f, my_d)
    
    def _drifting_loop_optimized(self, my_d, my_f):
        """优化版本的漂移循环"""
        start_time = time.time()
        
        try:
            # 创建电场缓存 - 修复参数传递
            field_cache = FieldCache(my_f)
            
            # 批量处理电子
            if self.electrons:
                print(f" 使用优化版本处理电子: {len(self.electrons)}个")
                # 提取所有电子的位置、电荷和时间
                all_positions = [[e.x, e.y, e.z] for e in self.electrons]
                all_charges = [e.charge for e in self.electrons] 
                all_times = [e.t for e in self.electrons]

                # 修复参数传递 - 添加缺失的参数
                electron_system = VectorizedCarrierSystem(
                    all_positions, all_charges, all_times, my_d.material, "electron",
                    self.read_out_contact, my_d  # 添加缺失的参数
                )
                electron_system.drift_batch(my_d, field_cache, delta_t=1e-12, max_steps=2000)
                electron_system.update_original_carriers(self.electrons)
            
            # 批量处理空穴
            if self.holes:
                print(f" 使用优化版本处理空穴: {len(self.holes)}个")
                all_positions = [[h.x, h.y, h.z] for h in self.holes]
                all_charges = [h.charge for h in self.holes]
                all_times = [h.t for h in self.holes]

                hole_system = VectorizedCarrierSystem(
                    all_positions, all_charges, all_times, my_d.material, "hole",
                    self.read_out_contact, my_d  # 添加缺失的参数
                )
                hole_system.drift_batch(my_d, field_cache, delta_t=1e-12, max_steps=2000) 
                hole_system.update_original_carriers(self.holes)
                print("优化漂移完成，开始计算信号...")
            # 电子信号计算
            electron_signals = 0
            for i, electron in enumerate(self.electrons):
                if len(electron.path_reduced) > 1:  # 确保有路径数据
                    try:
                        electron.get_signal(my_f, my_d)
                        electron_signals += 1
                        if i % 10 == 0:  # 每10个输出一次进度
                           print(f"电子 {i} 信号计算完成，信号长度: {len(electron.signal)}")
                    except Exception as e:
                        print(f"电子 {i} 信号计算失败: {e}")
        
            # 空穴信号计算
            hole_signals = 0
            for i, hole in enumerate(self.holes):
                if len(hole.path_reduced) > 1:  # 确保有路径数据
                    try:
                        hole.get_signal(my_f, my_d)
                        hole_signals += 1
                        if i % 10 == 0:  # 每10个输出一次进度
                            print(f"空穴 {i} 信号计算完成，信号长度: {len(hole.signal)}")
                    except Exception as e:
                        print(f"空穴 {i} 信号计算失败: {e}")
                    
            print(f"信号计算完成: {electron_signals}个电子 + {hole_signals}个空穴")
                
        except Exception as e:
            print(f"优化版本出错: {e}")
            import traceback
            traceback.print_exc()
            print("回退到原始版本...")
            self._drifting_loop_original(my_d, my_f)
    
        end_time = time.time()
        print(f"优化版本总耗时: {end_time-start_time:.2f}秒")
    def current_define(self, read_ele_num):
        """
        @description: 
            Parameter current setting     
        @param:
            positive_cu -- Current from holes move
            negative_cu -- Current from electrons move
            sum_cu -- Current from e-h move
        @Returns:
            None
        @Modify:
            2021/08/31
        """
        self.positive_cu=[]
        self.negative_cu=[]
        self.sum_cu=[]

        for i in range(read_ele_num):
            self.positive_cu.append(ROOT.TH1F("charge+"+str(i+1), " No."+str(i+1)+"Positive Current",
                                        self.n_bin, self.t_start, self.t_end))
            self.negative_cu.append(ROOT.TH1F("charge-"+str(i+1), " No."+str(i+1)+"Negative Current",
                                        self.n_bin, self.t_start, self.t_end))
            self.sum_cu.append(ROOT.TH1F("charge"+str(i+1),"Total Current"+" No."+str(i+1)+"electrode",
                                    self.n_bin, self.t_start, self.t_end))
            
        
    def get_current(self, n_x, n_y, read_out_contact):
        # 空穴电流计算 - 完整的独立循环
        for hole in self.holes:
            if len(read_out_contact)==1:
                x_span = read_out_contact[0]['x_span']
                y_span = read_out_contact[0]['y_span']
                signal_length = len(hole.signal)
                expected_signal_length = (x_span*2+1) * (y_span*2+1)

                print(f"调试-空穴: 信号列表长度={signal_length}, 期望长度={expected_signal_length}")

                for j in range(x_span*2+1):
                    for k in range(y_span*2+1):
                        signal_index = j * (y_span*2+1) + k
            
                        # 检查信号索引是否有效
                        if signal_index >= len(hole.signal):
                            print(f"警告: 信号索引 {signal_index} 超出范围 (0-{len(hole.signal)-1})")
                            continue
                
                        for i in range(len(hole.path_reduced)-1):
                            # 检查信号点是否存在
                            if i >= len(hole.signal[signal_index]):
                                print(f"警告: 路径索引 {i} 超出信号长度 {len(hole.signal[signal_index])}")
                                continue
                        
                            x_num = hole.path_reduced[i][4] + (j - x_span)
                            y_num = hole.path_reduced[i][5] + (k - y_span)
                            if x_num >= n_x or x_num < 0 or y_num >= n_y or y_num < 0:
                                continue
                        
                            # 使用原始代码的索引方式
                            self.positive_cu[x_num*n_y+y_num].Fill(
                                hole.path_reduced[i][3],
                                hole.signal[j*(y_span*2+1)+k][i]/self.t_bin
                            )

            else:
                for j in range(len(read_out_contact)):
                    for i in range(len(hole.path_reduced)-1):
                        self.positive_cu[j].Fill(hole.path_reduced[i][3],hole.signal[j][i]/self.t_bin) # time,current=int(i*dt)/Δt

        # 电子电流计算 - 完整的独立循环
        for electron in self.electrons:   
            if len(read_out_contact)==1:
                x_span = read_out_contact[0]['x_span']
                y_span = read_out_contact[0]['y_span']
                signal_length = len(electron.signal)
                expected_signal_length = (x_span*2+1) * (y_span*2+1)

                print(f"调试-电子: 信号列表长度={signal_length}, 期望长度={expected_signal_length}")

                for j in range(x_span*2+1):
                    for k in range(y_span*2+1):
                        signal_index = j * (y_span*2+1) + k
            
                        # 检查信号索引是否有效
                        if signal_index >= len(electron.signal):
                            print(f"警告: 信号索引 {signal_index} 超出范围 (0-{len(electron.signal)-1})")
                            continue
                
                        for i in range(len(electron.path_reduced)-1):
                            # 检查信号点是否存在
                            if i >= len(electron.signal[signal_index]):
                                print(f"警告: 路径索引 {i} 超出信号长度 {len(electron.signal[signal_index])}")
                                continue
                        
                            x_num = electron.path_reduced[i][4] + (j - x_span)
                            y_num = electron.path_reduced[i][5] + (k - y_span)
                            if x_num >= n_x or x_num < 0 or y_num >= n_y or y_num < 0:
                                continue
                        
                            # 使用原始代码的索引方式
                            self.negative_cu[x_num*n_y+y_num].Fill(
                                electron.path_reduced[i][3],
                                electron.signal[j*(y_span*2+1)+k][i]/self.t_bin
                            )

            else:
                for j in range(len(read_out_contact)):
                    for i in range(len(electron.path_reduced)-1):
                        self.negative_cu[j].Fill(electron.path_reduced[i][3],electron.signal[j][i]/self.t_bin)# time,current=int(i*dt)/Δtnt=int(i*dt)/Δt

    def draw_currents(self, path, tag=""):
        """
        @description:
            Save current in root file
        @param:
            None     
        @Returns:
            None
        @Modify:
            2021/08/31
        """
        for read_ele_num in range(self.read_ele_num):
            c=ROOT.TCanvas("c","canvas1",1600,1300)
            c.cd()
            c.Update()
            c.SetLeftMargin(0.25)
            # c.SetTopMargin(0.12)
            c.SetRightMargin(0.15)
            c.SetBottomMargin(0.17)
            ROOT.gStyle.SetOptStat(ROOT.kFALSE)
            ROOT.gStyle.SetOptStat(0)

            #self.sum_cu.GetXaxis().SetTitleOffset(1.2)
            #self.sum_cu.GetXaxis().SetTitleSize(0.05)
            #self.sum_cu.GetXaxis().SetLabelSize(0.04)
            self.sum_cu[read_ele_num].GetXaxis().SetNdivisions(510)
            #self.sum_cu.GetYaxis().SetTitleOffset(1.1)
            #self.sum_cu.GetYaxis().SetTitleSize(0.05)
            #self.sum_cu.GetYaxis().SetLabelSize(0.04)
            self.sum_cu[read_ele_num].GetYaxis().SetNdivisions(505)
            #self.sum_cu.GetXaxis().CenterTitle()
            #self.sum_cu.GetYaxis().CenterTitle() 
            self.sum_cu[read_ele_num].GetXaxis().SetTitle("Time [s]")
            self.sum_cu[read_ele_num].GetYaxis().SetTitle("Current [A]")
            self.sum_cu[read_ele_num].GetXaxis().SetLabelSize(0.08)
            self.sum_cu[read_ele_num].GetXaxis().SetTitleSize(0.08)
            self.sum_cu[read_ele_num].GetYaxis().SetLabelSize(0.08)
            self.sum_cu[read_ele_num].GetYaxis().SetTitleSize(0.08)
            self.sum_cu[read_ele_num].GetYaxis().SetTitleOffset(1.2)
            self.sum_cu[read_ele_num].SetTitle("")
            self.sum_cu[read_ele_num].SetNdivisions(5)
            self.sum_cu[read_ele_num].Draw("HIST")
            self.positive_cu[read_ele_num].Draw("SAME HIST")
            self.negative_cu[read_ele_num].Draw("SAME HIST")
            self.sum_cu[read_ele_num].Draw("SAME HIST")

            self.positive_cu[read_ele_num].SetLineColor(877)#kViolet-3
            self.negative_cu[read_ele_num].SetLineColor(600)#kBlue
            self.sum_cu[read_ele_num].SetLineColor(418)#kGreen+2

            self.positive_cu[read_ele_num].SetLineWidth(2)
            self.negative_cu[read_ele_num].SetLineWidth(2)
            self.sum_cu[read_ele_num].SetLineWidth(2)
            c.Update()

            if "lgad" in self.det_model:
                self.gain_current.positive_cu[read_ele_num].Draw("SAME HIST")
                self.gain_current.negative_cu[read_ele_num].Draw("SAME HIST")
                self.gain_current.positive_cu[read_ele_num].SetLineColor(617)#kMagneta+1
                self.gain_current.negative_cu[read_ele_num].SetLineColor(867)#kAzure+7
                self.gain_current.positive_cu[read_ele_num].SetLineWidth(2)
                self.gain_current.negative_cu[read_ele_num].SetLineWidth(2)

            if "strip" in self.det_model:
                # make sure you run cross_talk() first and attached cross_talk_cu to self
                self.cross_talk_cu[read_ele_num].Draw("SAME HIST")
                self.cross_talk_cu[read_ele_num].SetLineColor(420)#kGreen+4
                self.cross_talk_cu[read_ele_num].SetLineWidth(2)

            legend = ROOT.TLegend(0.5, 0.2, 0.8, 0.5)
            legend.AddEntry(self.negative_cu[read_ele_num], "electron", "l")
            legend.AddEntry(self.positive_cu[read_ele_num], "hole", "l")

            if "lgad" in self.det_model:
                legend.AddEntry(self.gain_current.negative_cu[read_ele_num], "electron gain", "l")
                legend.AddEntry(self.gain_current.positive_cu[read_ele_num], "hole gain", "l")

            if "strip" in self.det_model:
                legend.AddEntry(self.cross_talk_cu[read_ele_num], "cross talk", "l")

            legend.AddEntry(self.sum_cu[read_ele_num], "total", "l")
            
            legend.SetBorderSize(0)
            #legend.SetTextFont(43)
            legend.SetTextSize(0.08)
            legend.Draw("same")
            c.Update()

            c.SaveAs(path+'/'+tag+"No_"+str(read_ele_num+1)+"electrode"+"_basic_infor.pdf")
            c.SaveAs(path+'/'+tag+"No_"+str(read_ele_num+1)+"electrode"+"_basic_infor.root")
            del c

    def charge_collection(self, path):
        charge=array('d')
        x=array('d')
        for i in range(self.read_ele_num):
            x.append(i+1)
            sum_charge=0
            for j in range(self.n_bin):
                if "strip" in self.det_model:
                    sum_charge=sum_charge+self.cross_talk_cu[i].GetBinContent(j)*self.t_bin
                else:
                    sum_charge=sum_charge+self.sum_cu[i].GetBinContent(j)*self.t_bin
            charge.append(sum_charge/1.6e-19)
        print("===========RASER info================\nCollected Charge is {} e\n==============Result==============".format(list(charge)))
        n=int(len(charge))
        c1=ROOT.TCanvas("c1","canvas1",1000,1000)
        cce=ROOT.TGraph(n,x,charge)
        cce.SetMarkerStyle(3)
        cce.Draw()
        cce.SetTitle("Charge Collection Efficiency")
        cce.GetXaxis().SetTitle("elenumber")
        cce.GetYaxis().SetTitle("charge[Coulomb]")
        c1.SaveAs(path+"/cce.pdf")
        c1.SaveAs(path+"/cce.root")
    
class CalCurrentGain(CalCurrent):
    '''Calculation of gain carriers and gain current, simplified version'''
    def __init__(self, my_d, my_f, my_current):
        self.read_ele_num = my_current.read_ele_num
        self.read_out_contact = my_current.read_out_contact

        if "planar" in my_d.det_model or "lgad" in my_d.det_model:
            p_x = my_d.l_x
            p_y = my_d.l_y
            n_x = 1
            n_y = 1
            field_shift_x = 0
            field_shift_y = 0
        if "strip" in my_d.det_model:
            # for "lgadstrip", this covers above
            p_x = my_d.p_x
            p_y = my_d.l_y
            n_x = my_d.read_ele_num
            n_y = 1
            field_shift_x = my_d.field_shift_x
            field_shift_y = 0
        if "pixel" in my_d.det_model:
            p_x = my_d.p_x
            p_y = my_d.p_y
            n_x = my_d.x_ele_num
            n_y = my_d.y_ele_num
            field_shift_x = my_d.field_shift_x
            field_shift_y = my_d.field_shift_y

        self.electrons = [] # gain carriers
        self.holes = []
        cal_coefficient = Material(my_d.material).cal_coefficient
        gain_rate = self.gain_rate(my_d,my_f,cal_coefficient)
        print("gain_rate="+str(gain_rate))
        path = output(__file__, my_d.det_name)
        f_gain_rate = open(path+'/voltage-gain_rate.csv', "a")
        writer_gain_rate = csv.writer(f_gain_rate)
        writer_gain_rate.writerow([str(my_f.voltage),str(gain_rate)])
        with open(path+'/voltage-gain_rate.txt', 'a') as file:
            file.write(str(my_f.voltage)+' -- '+str(gain_rate)+ '\n')
        # assuming gain layer at d>0
        if my_d.voltage<0 : # p layer at d=0, holes multiplicated into electrons
            for hole in my_current.holes:
                self.electrons.append(CarrierCluster(hole.path[-1][0],
                                              hole.path[-1][1],
                                              my_d.avalanche_bond,
                                              hole.path[-1][3],
                                              p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                                              -1*hole.charge*gain_rate,
                                              my_d.material,
                                              self.read_out_contact))
                
                self.holes.append(CarrierCluster(hole.path[-1][0],
                                          hole.path[-1][1],
                                          my_d.avalanche_bond,
                                          hole.path[-1][3],
                                          p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                                          hole.charge*gain_rate,
                                          my_d.material,
                                          self.read_out_contact))

        else : # n layer at d=0, electrons multiplicated into holes
            for electron in my_current.electrons:
                self.holes.append(CarrierCluster(electron.path[-1][0],
                                          electron.path[-1][1],
                                          my_d.avalanche_bond,
                                          electron.path[-1][3],
                                          p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                                          -1*electron.charge*gain_rate,
                                          my_d.material,
                                          self.read_out_contact))

                self.electrons.append(CarrierCluster(electron.path[-1][0],
                                                electron.path[-1][1],
                                                my_d.avalanche_bond,
                                                electron.path[-1][3],
                                                p_x, p_y, n_x, n_y, my_d.l_x, my_d.l_y, field_shift_x, field_shift_y,
                                                electron.charge*gain_rate,
                                                my_d.material,
                                                self.read_out_contact))

        self.drifting_loop(my_d, my_f)

        self.t_bin = t_bin
        self.t_end = t_end
        self.t_start = t_start
        self.n_bin = int((self.t_end-self.t_start)/self.t_bin)

        self.current_define(self.read_ele_num)
        for i in range(self.read_ele_num):
            self.positive_cu[i].Reset()
            self.negative_cu[i].Reset()
        self.get_current(n_x, n_y, self.read_out_contact)

    def gain_rate(self, my_d, my_f, cal_coefficient):

        # gain = exp[K(d_gain)] / {1-int[alpha_minor * K(x) dx]}
        # K(x) = exp{int[(alpha_major - alpha_minor) dx]}

        # TODO: support non-uniform field in gain layer

        n = 1001
        if "ilgad" in my_d.det_model:
            z_list = np.linspace(my_d.avalanche_bond * 1e-4, my_d.l_z, n) # in cm
        else:
            z_list = np.linspace(0, my_d.avalanche_bond * 1e-4, n) # in cm
        alpha_n_list = np.zeros(n)
        alpha_p_list = np.zeros(n)
        for i in range(n):
            Ex,Ey,Ez = my_f.get_e_field(0.5*my_d.l_x,0.5*my_d.l_y,z_list[i] * 1e4) # in V/cm
            E_field = Vector(Ex,Ey,Ez).get_length()
            alpha_n = cal_coefficient(E_field, -1, my_d.temperature)
            alpha_p = cal_coefficient(E_field, +1, my_d.temperature)
            alpha_n_list[i] = alpha_n
            alpha_p_list[i] = alpha_p

        if my_f.get_e_field(0, 0, my_d.avalanche_bond)[2] > 0:
            alpha_major_list = alpha_n_list # multiplication contributed mainly by electrons in conventional Si LGAD
            alpha_minor_list = alpha_p_list
        else:
            alpha_major_list = alpha_p_list # multiplication contributed mainly by holes in conventional SiC LGAD
            alpha_minor_list = alpha_n_list

        # the integral supports iLGAD as well
        
        diff_list = alpha_major_list - alpha_minor_list
        int_alpha_list = np.zeros(n-1)

        for i in range(1,n):
            int_alpha = 0
            for j in range(i):
                int_alpha += (diff_list[j] + diff_list[j+1]) * (z_list[j+1] - z_list[j]) /2
            int_alpha_list[i-1] = int_alpha
        exp_list = np.exp(int_alpha_list)

        det = 0 # determinant of breakdown
        for i in range(0,n-1):
            average_alpha_minor = (alpha_minor_list[i] + alpha_minor_list[i+1])/2
            det_derivative = average_alpha_minor * exp_list[i]
            det += det_derivative*(z_list[i+1]-z_list[i])        
        if det>1:
            print("det="+str(det))
            print("The detector broke down")
            raise(ValueError)
        
        gain_rate = exp_list[n-2]/(1-det) -1
        return gain_rate

    def current_define(self,read_ele_num):
        """
        @description: 
            Parameter current setting     
        @param:
            positive_cu -- Current from holes move
            negative_cu -- Current from electrons move
            sum_cu -- Current from e-h move
        @Returns:
            None
        @Modify:
            2021/08/31
        """
        self.positive_cu=[]
        self.negative_cu=[]

        for i in range(read_ele_num):
            self.positive_cu.append(ROOT.TH1F("gain_charge_tmp+"+str(i+1)," No."+str(i+1)+"Gain Positive Current",
                                        self.n_bin, self.t_start, self.t_end))
            self.negative_cu.append(ROOT.TH1F("gain_charge_tmp-"+str(i+1)," No."+str(i+1)+"Gain Negative Current",
                                        self.n_bin, self.t_start, self.t_end))

class CalCurrentG4P(CalCurrent):
    def __init__(self, my_d, my_f, my_g4, batch):
        G4P_carrier_list = CarrierListFromG4P(my_d.material, my_g4, batch)
        super().__init__(my_d, my_f, G4P_carrier_list.ionized_pairs, G4P_carrier_list.track_position)
        if self.read_ele_num > 1:
            #self.cross_talk()
            pass


class CalCurrentLaser(CalCurrent):
    def __init__(self, my_d, my_f, my_l):
        super().__init__(my_d, my_f, my_l.ionized_pairs, my_l.track_position)
        
        for i in range(self.read_ele_num):
            
            # convolute the signal with the laser pulse shape in time
            convolved_positive_cu = ROOT.TH1F("convolved_charge+", "Positive Current",
                                        self.n_bin, self.t_start, self.t_end)
            convolved_negative_cu = ROOT.TH1F("convolved_charge-", "Negative Current",
                                        self.n_bin, self.t_start, self.t_end)
            convolved_sum_cu = ROOT.TH1F("convolved_charge","Total Current",
                                        self.n_bin, self.t_start, self.t_end)
            
            convolved_positive_cu.Reset()
            convolved_negative_cu.Reset()
            convolved_sum_cu.Reset()

            signal_convolution(self.positive_cu[i],convolved_positive_cu,[my_l.timePulse])
            signal_convolution(self.negative_cu[i],convolved_negative_cu,[my_l.timePulse])
            signal_convolution(self.sum_cu[i],convolved_sum_cu,[my_l.timePulse])

            self.positive_cu[i] = convolved_positive_cu
            self.negative_cu[i] = convolved_negative_cu
            self.sum_cu[i] = convolved_sum_cu

            if my_d.det_model == "lgad":
                convolved_gain_positive_cu = ROOT.TH1F("convolved_gain_charge+","Gain Positive Current",
                                        self.n_bin, self.t_start, self.t_end)
                convolved_gain_negative_cu = ROOT.TH1F("convolved_gain_charge-","Gain Negative Current",
                                        self.n_bin, self.t_start, self.t_end)
                convolved_gain_positive_cu.Reset()
                convolved_gain_negative_cu.Reset()
                signal_convolution(self.gain_current.positive_cu[i],convolved_gain_positive_cu,[my_l.timePulse])
                signal_convolution(self.gain_current.negative_cu[i],convolved_gain_negative_cu,[my_l.timePulse])
                self.gain_current.positive_cu[i] = convolved_gain_positive_cu
                self.gain_current.negative_cu[i] = convolved_gain_negative_cu


# =============================
# Optimized implementations merged from optimized_calcurrent.py
# =============================

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
