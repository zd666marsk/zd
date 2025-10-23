# -*- encoding: utf-8 -*-
"""
优化版本的电流计算模块 - 只修复迁移率计算，保持其他优化
"""

import numpy as np
import time
import math
import random
from .model import Material  # 导入Material类

class FieldCache:
    """电场缓存类 - 减少电场计算次数"""
    
    def __init__(self, my_f, resolution=1.0):
        self.my_f = my_f
        self.resolution = resolution
        self.e_field_cache = {}
        self.doping_cache = {}  # 添加掺杂缓存
        self._cache_stats = {'hits': 0, 'misses': 0}
        print(f"电场缓存初始化完成，分辨率: {resolution} um")
    
    def get_e_field_cached(self, x, y, z):
        """获取缓存的电场值"""
        try:
            key = (round(x/self.resolution), round(y/self.resolution), round(z/self.resolution))
            
            if key in self.e_field_cache:
                self._cache_stats['hits'] += 1
                return self.e_field_cache[key]
            else:
                self._cache_stats['misses'] += 1
                e_field = self.my_f.get_e_field(x, y, z)
                self.e_field_cache[key] = e_field
                return e_field
        except Exception as e:
            return self.my_f.get_e_field(x, y, z)
    
    def get_doping_cached(self, x, y, z):
        """获取掺杂浓度值"""
        try:
            key = (round(x/self.resolution), round(y/self.resolution), round(z/self.resolution))
            if key in self.doping_cache:
                return self.doping_cache[key]
            else:
                doping = self.my_f.get_doping(x, y, z)
                self.doping_cache[key] = doping
                return doping
        except Exception as e:
            return self.my_f.get_doping(x, y, z)

class VectorizedCarrierSystem:
    """向量化载流子系统 - 只修复迁移率计算"""
    
    def __init__(self, all_positions, all_charges, all_times, material, carrier_type="electron", 
                 read_out_contact=None, my_d=None):
        self.positions = np.array(all_positions, dtype=np.float64)
        self.charges = np.array(all_charges, dtype=np.float64)
        self.times = np.array(all_times, dtype=np.float64)
        self.active = np.ones(len(all_charges), dtype=bool)
        self.end_conditions = np.zeros(len(all_charges), dtype=np.int8)
        self.steps_drifted = np.zeros(len(all_charges), dtype=np.int32)
        self.carrier_type = carrier_type
        self.read_out_contact = read_out_contact
        self.my_d = my_d
        
        # 关键修复：创建Material对象
        self.material = Material(material)
        
        # 存储完整路径
        self.paths = [[] for _ in range(len(all_charges))]
        for i in range(len(all_charges)):
            self.paths[i].append([all_positions[i][0], all_positions[i][1], all_positions[i][2], all_times[i]])
        
        # 探测器参数
        self.detector_params = self._extract_detector_params(my_d)
        
        # 物理常数
        self.kboltz = 8.617385e-5
        self.e0 = 1.60217733e-19
        
        print(f"向量化系统初始化: {len(all_charges)}个{carrier_type}")
    
    def _extract_detector_params(self, my_d):
        """提取探测器参数"""
        params = {}
        try:
            if my_d:
                params['l_x'] = getattr(my_d, 'l_x', 100.0)
                params['l_y'] = getattr(my_d, 'l_y', 100.0)
                params['p_x'] = getattr(my_d, 'p_x', 1.0)
                params['p_y'] = getattr(my_d, 'p_y', 1.0)
                params['n_x'] = getattr(my_d, 'x_ele_num', 1)
                params['n_y'] = getattr(my_d, 'y_ele_num', 1)
                params['field_shift_x'] = getattr(my_d, 'field_shift_x', 0)
                params['field_shift_y'] = getattr(my_d, 'field_shift_y', 0)
            else:
                params.update({
                    'l_x': 100.0, 'l_y': 100.0,
                    'p_x': 1.0, 'p_y': 1.0,
                    'n_x': 1, 'n_y': 1,
                    'field_shift_x': 0, 'field_shift_y': 0
                })
        except Exception as e:
            params.update({
                'l_x': 100.0, 'l_y': 100.0,
                'p_x': 1.0, 'p_y': 1.0,
                'n_x': 1, 'n_y': 1,
                'field_shift_x': 0, 'field_shift_y': 0
            })
        return params
    
    def drift_batch(self, my_d, field_cache, delta_t=1e-12, max_steps=2000):
        """批量漂移主函数"""
        print(f"开始批量漂移{self.carrier_type}，最多{max_steps}步，时间步长{delta_t}s")
        
        start_time = time.time()
        steps_completed = 0
        
        for step in range(max_steps):
            n_terminated = self.drift_step_batch(my_d, field_cache, delta_t)
            steps_completed = step + 1
            
            if step % 100 == 0:
                stats = self.get_statistics()
                print(f"  步骤 {step}: {stats['active_carriers']}个活跃载流子")
            
            if not np.any(self.active):
                print("所有载流子停止漂移")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        final_stats = self.get_statistics()
        print(f"批量漂移完成: 共{steps_completed}步，耗时{total_time:.2f}秒")
        print(f"最终状态: {final_stats['active_carriers']}个活跃，平均步数{final_stats['average_steps']:.1f}")
        
        return True
    
    def drift_step_batch(self, my_d, field_cache, delta_t=1e-12):
        """批量单步漂移计算 - 只修复迁移率计算"""
        if not np.any(self.active):
            return 0
            
        n_terminated = 0
        
        for idx in range(len(self.active)):
            if not self.active[idx]:
                continue
                
            x, y, z = self.positions[idx]
            charge = self.charges[idx]
            
            try:
                e_field = field_cache.get_e_field_cached(x, y, z)
                Ex, Ey, Ez = e_field
            except Exception as e:
                self.active[idx] = False
                n_terminated += 1
                continue
            
            intensity = math.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)
            
            if intensity <= 1.0:
                self.active[idx] = False
                n_terminated += 1
                continue
            
            # 关键修复：使用正确的迁移率计算
            doping = field_cache.get_doping_cached(x, y, z)
            mu = self.material.cal_mobility(my_d.temperature, doping, charge, intensity)
            
            # 其余计算保持不变
            if charge > 0:
                vx = Ex * mu
                vy = Ey * mu
                vz = Ez * mu
            else:
                vx = -Ex * mu
                vy = -Ey * mu
                vz = -Ez * mu
            
            delta_x = vx * delta_t * 1e4
            delta_y = vy * delta_t * 1e4
            delta_z = vz * delta_t * 1e4
            
            try:
                diffusion = math.sqrt(2.0 * self.kboltz * mu * my_d.temperature * delta_t) * 1e4
                dif_x = random.gauss(0.0, diffusion)
                dif_y = random.gauss(0.0, diffusion)
                dif_z = random.gauss(0.0, diffusion)
            except:
                dif_x, dif_y, dif_z = 0.0, 0.0, 0.0
            
            new_x = x + delta_x + dif_x
            new_y = y + delta_y + dif_y
            new_z = z + delta_z + dif_z
            
            self.positions[idx] = [new_x, new_y, new_z]
            self.times[idx] += delta_t
            self.steps_drifted[idx] += 1
            
            self.paths[idx].append([new_x, new_y, new_z, self.times[idx]])
            
            try:
                if (new_x <= 0 or new_x >= my_d.l_x or 
                    new_y <= 0 or new_y >= my_d.l_y or 
                    new_z <= 0 or new_z >= my_d.l_z):
                    self.active[idx] = False
                    n_terminated += 1
                elif self.times[idx] > 10e-9:
                    self.active[idx] = False
                    n_terminated += 1
            except Exception as e:
                self.active[idx] = False
                n_terminated += 1
        
        return n_terminated
    
    # 其余方法保持不变...
    def get_statistics(self):
        n_total = len(self.active)
        n_active = np.sum(self.active)
        
        if np.any(self.steps_drifted > 0):
            avg_steps = np.mean(self.steps_drifted[self.steps_drifted > 0])
        else:
            avg_steps = 0
            
        return {
            'total_carriers': n_total,
            'active_carriers': n_active,
            'average_steps': avg_steps,
            'max_steps': np.max(self.steps_drifted),
            'carrier_type': self.carrier_type
        }
    
    def update_original_carriers(self, original_carriers):
        print(f"更新{self.carrier_type}状态...")
        updated_count = 0
        for i, carrier in enumerate(original_carriers):
            if i < len(self.positions):
                try:
                    carrier.x = float(self.positions[i][0])
                    carrier.y = float(self.positions[i][1])
                    carrier.z = float(self.positions[i][2])
                    carrier.t = float(self.times[i])
                
                    carrier.path = []
                    for point in self.paths[i]:
                        carrier.path.append([float(point[0]), float(point[1]), float(point[2]), float(point[3])])
                
                    self._rebuild_carrier_attributes(carrier, i)
                    self._reinitialize_signal_list(carrier)
                
                    if not self.active[i]:
                       carrier.end_condition = "批量漂移结束"
                
                    updated_count += 1
                except Exception as e:
                    print(f"更新载流子 {i} 时出错: {e}")
    
        print(f"已更新 {updated_count} 个{self.carrier_type}")
        return updated_count

    def _rebuild_carrier_attributes(self, carrier, idx):
        try:
            params = self.detector_params
        
            for key, value in params.items():
                setattr(carrier, key, value)
        
            if (hasattr(carrier, 'read_out_contact') and carrier.read_out_contact and 
                len(carrier.read_out_contact) == 1 and
                (carrier.read_out_contact[0].get('x_span', 0) != 0 or 
                 carrier.read_out_contact[0].get('y_span', 0) != 0)):
            
                carrier.x_reduced = (carrier.x - params['l_x']/2) % params['p_x'] + params['field_shift_x']
                carrier.y_reduced = (carrier.y - params['l_y']/2) % params['p_y'] + params['field_shift_y']
            else:
                carrier.x_reduced = carrier.x
                carrier.y_reduced = carrier.y
        
            carrier.x_num = int((carrier.x - params['l_x']/2) // params['p_x'] + params['n_x']/2.0)
            carrier.y_num = int((carrier.y - params['l_y']/2) // params['p_y'] + params['n_y']/2.0)
        
            carrier.path_reduced = []
            for j, point in enumerate(self.paths[idx]):
                x, y, z, t = point[0], point[1], point[2], point[3]
            
                if (hasattr(carrier, 'read_out_contact') and carrier.read_out_contact and 
                    len(carrier.read_out_contact) == 1 and
                    (carrier.read_out_contact[0].get('x_span', 0) != 0 or 
                     carrier.read_out_contact[0].get('y_span', 0) != 0)):
                
                    x_reduced = (x - params['l_x']/2) % params['p_x'] + params['field_shift_x']
                    y_reduced = (y - params['l_y']/2) % params['p_y'] + params['field_shift_y']
                else:
                    x_reduced = x
                    y_reduced = y
            
                x_num = int((x - params['l_x']/2) // params['p_x'] + params['n_x']/2.0)
                y_num = int((y - params['l_y']/2) // params['p_y'] + params['n_y']/2.0)
            
                carrier.path_reduced.append([x_reduced, y_reduced, z, t, x_num, y_num])
            
        except Exception as e:
            carrier.path_reduced = [[carrier.x, carrier.y, carrier.z, carrier.t, 0, 0]]
    
    def _reinitialize_signal_list(self, carrier):
        try:
            if hasattr(carrier, 'read_out_contact') and carrier.read_out_contact:
                if len(carrier.read_out_contact) == 1:
                    x_span = carrier.read_out_contact[0].get('x_span', 0)
                    y_span = carrier.read_out_contact[0].get('y_span', 0)
                    carrier.signal = [[] for j in range((2*x_span+1)*(2*y_span+1))]
                else:
                    carrier.signal = [[] for j in range(len(carrier.read_out_contact))]
            else:
                carrier.signal = [[]]
        except Exception as e:
            carrier.signal = [[]]