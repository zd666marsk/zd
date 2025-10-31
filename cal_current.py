# -*- encoding: utf-8 -*-

'''
Description:  
    Simulate e-h pairs drifting and calculate induced current
@Date       : 2021/09/02 14:01:46
@Author     : Yuhang Tan, Chenxi Fu
@version    : 2.0
'''

import random
import math
import os
from array import array
import csv
import time
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)

from .model import Material
from ..interaction.carrier_list import CarrierListFromG4P
from ..util.math import Vector, signal_convolution
from ..util.output import output

OPTIMIZATION_AVAILABLE = False
FieldCache = None
VectorizedCarrierSystem = None

try:
    # 尝试直接导入（如果 optimized_calcurrent.py 在 Python 路径中）
    from optimized_calcurrent import FieldCache as FC1, VectorizedCarrierSystem as VCS1
    FieldCache = FC1
    VectorizedCarrierSystem = VCS1
    OPTIMIZATION_AVAILABLE = True
    print(" 优化模块加载成功 - 使用直接导入")
except ImportError as e1:
    try:
        # 尝试相对导入（如果 optimized_calcurrent.py 在同一包内）
        from .optimized_calcurrent import FieldCache as FC2, VectorizedCarrierSystem as VCS2
        FieldCache = FC2
        VectorizedCarrierSystem = VCS2
        OPTIMIZATION_AVAILABLE = True
        print(" 优化模块加载成功 - 使用相对导入")
    except ImportError as e2:
        try:
            # 尝试从上层目录导入
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from optimized_calcurrent import FieldCache as FC3, VectorizedCarrierSystem as VCS3
            FieldCache = FC3
            VectorizedCarrierSystem = VCS3
            OPTIMIZATION_AVAILABLE = True
            print(" 优化模块加载成功 - 使用路径导入")
        except ImportError as e3:
            OPTIMIZATION_AVAILABLE = False
            print(f" 优化模块加载失败 - 使用原始版本")
            print(f"   直接导入错误: {e1}")
            print(f"   相对导入错误: {e2}")
            print(f"   路径导入错误: {e3}")

# 如果所有导入都失败，创建空类避免后续错误
if not OPTIMIZATION_AVAILABLE:
    class FieldCache:
        def __init__(self, *args, **kwargs):
            raise ImportError("优化模块未加载")
    
    class VectorizedCarrierSystem:
        def __init__(self, *args, **kwargs):
            raise ImportError("优化模块未加载")
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
    def __init__(self, x_init, y_init, z_init, t_init, p_x, p_y, n_x, n_y, l_x, l_y, field_shift_x, field_shift_y, charge, material, weighting_field):
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

        # Since the signal amplitude is proportional to charge:
        # - For n charge carriers (each with charge q) undergoing random walks with diffusion coefficient D,
        #   the variance of signal perturbation becomes n times that of a single charge carrier
        # - A single charge carrier with charge n*q under the same diffusion conditions
        #   also produces signal perturbation variance n times that of a single charge
        # This equivalence implies that a group of charge carriers can be treated as
        # a single composite carrier when modeling random walk behavior,
        # provided their total charge and diffusion characteristics are properly scaled

        kboltz=8.617385e-5 #eV/K
        diffusion = (2.0*kboltz*mu*my_d.temperature*delta_t)**0.5

        dif_x=random.gauss(0.0,diffusion)*1e4
        dif_y=random.gauss(0.0,diffusion)*1e4
        dif_z=random.gauss(0.0,diffusion)*1e4

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
