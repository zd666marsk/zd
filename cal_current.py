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

import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)

from .model import Material
from interaction.carrier_list import CarrierListFromG4P
from util.math import Vector, signal_convolution
from util.output import output

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

        self.path_reduced = [[self.x_reduced, self.y_reduced, z_init, t_init,
                              self.x_num, self.y_num]]

        if len(weighting_field) == 1 and (weighting_field[0]['x_span'] != 0 or weighting_field[0]['y_span'] != 0):
            self.signal = [[] for j in range((2*weighting_field[0]['x_span']+1)*(2*weighting_field[0]['y_span']+1))]
        else:
            self.signal = [[] for j in range(len(weighting_field))]
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
        for electron in self.electrons:
            while not electron.not_in_sensor(my_d) and not electron.not_in_field_range(my_d) and not electron.drift_end(my_f):
                electron.drift_single_step(my_d, my_f)
            electron.get_signal(my_f,my_d)
        for hole in self.holes:
            while not hole.not_in_sensor(my_d) and not hole.not_in_field_range(my_d) and not hole.drift_end(my_f):
                hole.drift_single_step(my_d, my_f)
            hole.get_signal(my_f,my_d)

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
        for hole in self.holes:
            if len(read_out_contact)==1:
                x_span = read_out_contact[0]['x_span']
                y_span = read_out_contact[0]['y_span']
                for j in range(x_span*2+1):
                    for k in range(y_span*2+1):
                        for i in range(len(hole.path_reduced)-1):
                            x_num = hole.path_reduced[i][4] + (j - x_span)
                            y_num = hole.path_reduced[i][5] + (k - y_span)
                            if x_num >= n_x or x_num < 0 or y_num >= n_y or y_num < 0:
                                continue
                            self.positive_cu[x_num*n_y+y_num].Fill(hole.path_reduced[i][3],hole.signal[j*(y_span*2+1)+k][i]/self.t_bin)
                            # time,current=int(i*dt)/Δtj][i]/self.t_bin)# time,current=int(i*dt)/Δt
            else:
                for j in range(len(read_out_contact)):
                    for i in range(len(hole.path_reduced)-1):
                        self.positive_cu[j].Fill(hole.path_reduced[i][3],hole.signal[j][i]/self.t_bin) # time,current=int(i*dt)/Δt

        for electron in self.electrons:   
            if len(read_out_contact)==1:
                x_span = read_out_contact[0]['x_span']
                y_span = read_out_contact[0]['y_span']
                for j in range(x_span*2+1):
                    for k in range(y_span*2+1):
                        for i in range(len(electron.path_reduced)-1):
                            x_num = electron.path_reduced[i][4] + (j - x_span)
                            y_num = electron.path_reduced[i][5] + (k - y_span)
                            if x_num >= n_x or x_num < 0 or y_num >= n_y or y_num < 0:
                                continue
                            self.negative_cu[x_num*n_y+y_num].Fill(electron.path_reduced[i][3],electron.signal[j*(y_span*2+1)+k][i]/self.t_bin)
                            # time,current=int(i*dt)/Δtj][i]/self.t_bin)# time,current=int(i*dt)/Δt
            else:
                for j in range(len(read_out_contact)):
                    for i in range(len(electron.path_reduced)-1):
                        self.negative_cu[j].Fill(electron.path_reduced[i][3],electron.signal[j][i]/self.t_bin)# time,current=int(i*dt)/Δt

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

# 这些类保持在单文件中，便于需要时手动启用优化路径。
import math
import os
import random
import time
import logging

import numpy as np

try:
    import matplotlib.pyplot as plt  # 可选依赖，仅在生成可视化时使用
except ImportError:  # pragma: no cover - 可视化并非主流程必需
    plt = None

# 与主流程共用 Material，若作为独立模块导入时提供回退路径
try:  # pragma: no cover - 在包外直接运行时触发
    from .model import Material as _OptimizationMaterial
except ImportError:  # pragma: no cover - 允许脚本模式直接调用
    from model import Material as _OptimizationMaterial
else:
    # 成功的相对导入时，与主流程使用同一个 Material 引用
    pass

if 'Material' not in globals():  # pragma: no cover - 脚本直接调用
    Material = _OptimizationMaterial

logger = logging.getLogger(__name__ + ".optimization")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class FieldCache:
    """电场缓存层，来自优化版实现。"""

    def __init__(self, my_f, resolution=5.0):
        self.my_f = my_f
        self.resolution = resolution
        self.e_field_cache = {}
        self.doping_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        logger.debug("FieldCache initialized with resolution %.2f um", resolution)

    def get_e_field_cached(self, x, y, z):
        """按网格缓存电场。"""
        try:
            if not self._is_position_valid(x, y, z):
                return self._safe_get_e_field(x, y, z)

            key = (
                int(round(x / self.resolution)),
                int(round(y / self.resolution)),
                int(round(z / self.resolution)),
            )
            if key in self.e_field_cache:
                self._cache_stats['hits'] += 1
                return self.e_field_cache[key]

            self._cache_stats['misses'] += 1
            value = self._safe_get_e_field(x, y, z)
            if value is not None:
                self.e_field_cache[key] = value
            return value
        except Exception as exc:  # pragma: no cover - 容错逻辑
            self._cache_stats['errors'] += 1
            logger.warning("Field cache fallback at (%.1f, %.1f, %.1f): %s", x, y, z, exc)
            return self._safe_get_e_field(x, y, z)

    def get_doping_cached(self, x, y, z):
        """按网格缓存掺杂浓度。"""
        try:
            if not self._is_position_valid(x, y, z):
                return self._safe_get_doping(x, y, z)

            key = (
                int(round(x / self.resolution)),
                int(round(y / self.resolution)),
                int(round(z / self.resolution)),
            )
            if key in self.doping_cache:
                return self.doping_cache[key]

            value = self._safe_get_doping(x, y, z)
            if value is not None:
                self.doping_cache[key] = value
            return value
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.warning("Doping cache fallback at (%.1f, %.1f, %.1f): %s", x, y, z, exc)
            return 0.0

    def _is_position_valid(self, x, y, z):
        max_size = 50000  # 50 mm 宽泛范围，兼容大器件
        if (
            abs(x) > max_size
            or abs(y) > max_size
            or abs(z) > max_size
            or math.isnan(x)
            or math.isnan(y)
            or math.isnan(z)
            or math.isinf(x)
            or math.isinf(y)
            or math.isinf(z)
        ):
            return False
        return True

    def _safe_get_e_field(self, x, y, z):
        try:
            return self.my_f.get_e_field(x, y, z)
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.error("E-field lookup failed: %s", exc)
            return [0.0, 0.0, 100.0]

    def _safe_get_doping(self, x, y, z):
        try:
            return self.my_f.get_doping(x, y, z)
        except Exception as exc:  # pragma: no cover
            logger.warning("Doping lookup failed: %s", exc)
            return 0.0

    def get_cache_stats(self):
        total = self._cache_stats['hits'] + self._cache_stats['misses'] + self._cache_stats['errors']
        hit_rate = self._cache_stats['hits'] / total if total else 0.0
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'errors': self._cache_stats['errors'],
            'hit_rate': hit_rate,
            'total_entries': len(self.e_field_cache),
        }


class VectorizedCarrierSystem:
    """批量载流子漂移实现，直接从优化版迁移。"""

    def __init__(
        self,
        all_positions,
        all_charges,
        all_times,
        material,
        carrier_type="electron",
        read_out_contact=None,
        my_d=None,
    ):
        self._validate_inputs(all_positions, all_charges, all_times)

        self.positions = np.array(all_positions, dtype=np.float64)
        self.charges = np.array(all_charges, dtype=np.float64)
        self.times = np.array(all_times, dtype=np.float64)
        self.active = np.ones(len(all_charges), dtype=bool)
        self.end_conditions = np.zeros(len(all_charges), dtype=np.int8)
        self.steps_drifted = np.zeros(len(all_charges), dtype=np.int32)
        self.carrier_type = carrier_type
        self.read_out_contact = read_out_contact
        self.my_d = my_d

        self.material = self._create_material_safe(material)
        self.detector_params = self._extract_detector_params_robust(my_d)
        self._initialize_other_attributes(all_positions)

        self.kboltz = 8.617385e-5
        self.e0 = 1.60217733e-19

        self.performance_stats = {
            'total_steps': 0,
            'field_calculations': 0,
            'boundary_checks': 0,
            'carriers_terminated': 0,
            'low_field_terminations': 0,
            'boundary_terminations': 0,
        }

        logger.debug(
            "VectorizedCarrierSystem init: %d %s",
            len(all_charges),
            carrier_type,
        )

    def _validate_inputs(self, positions, charges, times):
        if len(positions) == 0:
            raise ValueError("载流子位置列表不能为空")
        if len(positions) != len(charges) or len(positions) != len(times):
            raise ValueError("位置、电荷和时间数组长度不一致")
        for i, pos in enumerate(positions):
            if len(pos) != 3:
                raise ValueError(f"位置数据 {i} 格式错误，应为 [x, y, z]")
            x, y, z = pos
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                raise ValueError(f"位置数据 {i} 包含 NaN 值")

    def _create_material_safe(self, material):
        try:
            return Material(material)
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.warning("Material(%s) fallback: %s", material, exc)
            try:
                return Material("si")
            except Exception:  # pragma: no cover - 兜底
                class FallbackMaterial:
                    name = "fallback_si"
                return FallbackMaterial()

    def _extract_detector_params_robust(self, my_d):
        params = {}
        try:
            if my_d is not None:
                params['l_x'] = getattr(my_d, 'l_x', 10000.0)
                params['l_y'] = getattr(my_d, 'l_y', 10000.0)
                params['l_z'] = getattr(my_d, 'l_z', 300.0)
                params['p_x'] = getattr(my_d, 'p_x', 50.0)
                params['p_y'] = getattr(my_d, 'p_y', 50.0)
                params['n_x'] = int(getattr(my_d, 'x_ele_num', 200))
                params['n_y'] = int(getattr(my_d, 'y_ele_num', 200))
                params['field_shift_x'] = getattr(my_d, 'field_shift_x', 0.0)
                params['field_shift_y'] = getattr(my_d, 'field_shift_y', 0.0)
                params['temperature'] = getattr(my_d, 'temperature', 300.0)
                params['boundary_tolerance'] = 1.0
                params['max_drift_time'] = 100e-9
                params['min_field_strength'] = 1.0
            else:
                params.update(self._get_large_detector_defaults())
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.error("Detector param extraction failed: %s", exc)
            params.update(self._get_large_detector_defaults())
        return params

    def _get_large_detector_defaults(self):
        return {
            'l_x': 10000.0,
            'l_y': 10000.0,
            'l_z': 300.0,
            'p_x': 50.0,
            'p_y': 50.0,
            'n_x': 200,
            'n_y': 200,
            'field_shift_x': 0.0,
            'field_shift_y': 0.0,
            'temperature': 300.0,
            'boundary_tolerance': 1.0,
            'max_drift_time': 100e-9,
            'min_field_strength': 1.0,
        }

    def _initialize_other_attributes(self, all_positions):
        self.reduced_positions = np.zeros((len(all_positions), 2), dtype=np.float64)
        for i, pos in enumerate(all_positions):
            x, y, _ = pos
            x_reduced, y_reduced = self._calculate_reduced_coords(x, y, self.my_d)
            self.reduced_positions[i] = [x_reduced, y_reduced]

        self.paths = [[] for _ in range(len(all_positions))]
        self.paths_reduced = [[] for _ in range(len(all_positions))]
        for i in range(len(all_positions)):
            x, y, z = all_positions[i]
            t = self.times[i]
            self.paths[i].append([x, y, z, t])

            x_reduced, y_reduced = self.reduced_positions[i]
            x_num, y_num = self._calculate_electrode_numbers(x, y)
            self.paths_reduced[i].append([x_reduced, y_reduced, z, t, x_num, y_num])

    def _calculate_reduced_coords(self, x, y, my_d):
        params = self.detector_params
        use_reduced = (
            self.read_out_contact
            and len(self.read_out_contact) == 1
            and (
                self.read_out_contact[0].get('x_span', 0) != 0
                or self.read_out_contact[0].get('y_span', 0) != 0
            )
        )
        if use_reduced:
            x_reduced = (x - params['l_x'] / 2) % params['p_x'] + params['field_shift_x']
            y_reduced = (y - params['l_y'] / 2) % params['p_y'] + params['field_shift_y']
        else:
            x_reduced = x
            y_reduced = y
        return x_reduced, y_reduced

    def _calculate_electrode_numbers(self, x, y):
        params = self.detector_params
        try:
            x_num = int((x - params['l_x'] / 2) // params['p_x'] + params['n_x'] / 2.0)
            y_num = int((y - params['l_y'] / 2) // params['p_y'] + params['n_y'] / 2.0)
            x_num = max(0, min(params['n_x'] - 1, x_num))
            y_num = max(0, min(params['n_y'] - 1, y_num))
            return x_num, y_num
        except Exception:  # pragma: no cover - 容错逻辑
            return params['n_x'] // 2, params['n_y'] // 2

    def _calculate_correct_mobility(self, temperature, doping, charge, electric_field):
        try:
            field_strength = np.linalg.norm(electric_field)
            if charge > 0:
                mu_low_field, beta, vsat = 480.0, 1.0, 0.95e7
            else:
                mu_low_field, beta, vsat = 1350.0, 2.0, 1.0e7
            if field_strength > 1e3:
                e0 = vsat / mu_low_field
                mu = mu_low_field / (1 + (field_strength / e0) ** beta) ** (1 / beta)
                mu = max(mu, vsat / field_strength)
            else:
                mu = mu_low_field
            return mu
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.warning("Mobility fallback: %s", exc)
            return 1350.0 if charge < 0 else 480.0

    def _check_boundary_conditions(self, x, y, z):
        params = self.detector_params
        tolerance = params['boundary_tolerance']
        return (
            x <= -tolerance
            or x >= params['l_x'] + tolerance
            or y <= -tolerance
            or y >= params['l_y'] + tolerance
            or z <= -tolerance
            or z >= params['l_z'] + tolerance
        )

    def drift_batch(self, my_d, field_cache, delta_t=1e-12, max_steps=5000):
        logger.info(
            "开始批量漂移%s，最多%d步，时间步长%gs",
            self.carrier_type,
            max_steps,
            delta_t,
        )
        start_time = time.time()
        delta_t_cm = delta_t * 1e4
        total_carriers = len(self.active)
        logger.info("初始状态: %d/%d 个活跃载流子", np.sum(self.active), total_carriers)

        for step in range(max_steps):
            if step % 100 == 0:
                self._log_progress(step, total_carriers)
            self.drift_step_batch(my_d, field_cache, delta_t, delta_t_cm, step)
            self.performance_stats['total_steps'] += 1
            if not np.any(self.active):
                logger.info("所有载流子停止漂移")
                break

        self._log_final_stats(start_time, max_steps)
        return True

    def drift_step_batch(self, my_d, field_cache, delta_t, delta_t_cm, step=0):
        if not np.any(self.active):
            return 0

        n_terminated = 0
        params = self.detector_params
        diffusion_constant = math.sqrt(2.0 * self.kboltz * params['temperature'] * delta_t) * 1e4

        for idx in range(len(self.active)):
            if not self.active[idx]:
                continue

            x, y, z = self.positions[idx]
            charge = self.charges[idx]

            self.performance_stats['boundary_checks'] += 1
            if self._check_boundary_conditions(x, y, z):
                self.active[idx] = False
                self.end_conditions[idx] = 1
                n_terminated += 1
                self.performance_stats['boundary_terminations'] += 1
                continue

            if self.times[idx] > params['max_drift_time']:
                self.active[idx] = False
                self.end_conditions[idx] = 4
                n_terminated += 1
                continue

            e_field = self._get_e_field_safe(field_cache, x, y, z, idx)
            if e_field is None:
                continue

            ex, ey, ez = e_field
            intensity = math.sqrt(ex * ex + ey * ey + ez * ez)
            if intensity <= params['min_field_strength']:
                self.active[idx] = False
                self.end_conditions[idx] = 3
                n_terminated += 1
                self.performance_stats['low_field_terminations'] += 1
                continue

            try:
                doping = field_cache.get_doping_cached(x, y, z)
                mu = self._calculate_correct_mobility(params['temperature'], doping, charge, e_field)
            except Exception:  # pragma: no cover - 容错逻辑
                mu = 1350.0 if charge < 0 else 480.0

            delta_x, delta_y, delta_z = self._calculate_displacement(charge, e_field, mu, delta_t_cm)
            dif_x, dif_y, dif_z = self._calculate_diffusion(diffusion_constant, mu)
            self._update_carrier_position(idx, delta_x, delta_y, delta_z, dif_x, dif_y, dif_z, delta_t)

        self.performance_stats['carriers_terminated'] += n_terminated
        return n_terminated

    def _get_e_field_safe(self, field_cache, x, y, z, idx):
        try:
            self.performance_stats['field_calculations'] += 1
            e_field = field_cache.get_e_field_cached(x, y, z)
            if e_field is None or len(e_field) != 3:
                raise ValueError("无效的电场值")
            return e_field
        except Exception as exc:  # pragma: no cover - 容错逻辑
            logger.warning("电场获取失败 (carrier %d): %s", idx, exc)
            self.active[idx] = False
            self.end_conditions[idx] = 2
            return None

    def _calculate_displacement(self, charge, e_field, mu, delta_t_cm):
        ex, ey, ez = e_field
        if charge > 0:
            vx, vy, vz = ex * mu, ey * mu, ez * mu
        else:
            vx, vy, vz = -ex * mu, -ey * mu, -ez * mu
        return vx * delta_t_cm, vy * delta_t_cm, vz * delta_t_cm

    def _calculate_diffusion(self, diffusion_constant, mu):
        try:
            diffusion = diffusion_constant * math.sqrt(mu)
            return (
                random.gauss(0.0, diffusion),
                random.gauss(0.0, diffusion),
                random.gauss(0.0, diffusion),
            )
        except Exception:  # pragma: no cover - 容错逻辑
            return 0.0, 0.0, 0.0

    def _update_carrier_position(self, idx, delta_x, delta_y, delta_z, dif_x, dif_y, dif_z, delta_t):
        x, y, z = self.positions[idx]
        new_x = x + delta_x + dif_x
        new_y = y + delta_y + dif_y
        new_z = z + delta_z + dif_z

        self.positions[idx] = [new_x, new_y, new_z]
        self.reduced_positions[idx] = self._calculate_reduced_coords(new_x, new_y, self.my_d)
        self.times[idx] += delta_t
        self.steps_drifted[idx] += 1

        self.paths[idx].append([new_x, new_y, new_z, self.times[idx]])
        x_reduced, y_reduced = self.reduced_positions[idx]
        x_num, y_num = self._calculate_electrode_numbers(new_x, new_y)
        self.paths_reduced[idx].append([x_reduced, y_reduced, new_z, self.times[idx], x_num, y_num])

    def _log_progress(self, step, total_carriers):
        active_count = np.sum(self.active)
        progress = (total_carriers - active_count) / total_carriers * 100 if total_carriers else 100
        logger.info("  步骤 %d: %d 个活跃载流子 (%.1f%% 完成)", step, active_count, progress)

    def _log_final_stats(self, start_time, max_steps):
        end_time = time.time()
        final_stats = self.get_statistics()
        perf_stats = self.get_performance_stats()
        logger.info(
            "批量漂移完成: 共%d步，耗时%.2f秒",
            self.performance_stats['total_steps'],
            end_time - start_time,
        )
        logger.info(
            "最终状态: %d 个活跃，平均步数 %.1f",
            final_stats['active_carriers'],
            final_stats['average_steps'],
        )
        logger.debug("性能统计: %s", perf_stats)

    def get_statistics(self):
        n_total = len(self.active)
        n_active = int(np.sum(self.active))
        if np.any(self.steps_drifted > 0):
            avg_steps = float(np.mean(self.steps_drifted[self.steps_drifted > 0]))
            max_steps = int(np.max(self.steps_drifted))
        else:
            avg_steps = 0.0
            max_steps = 0
        end_condition_counts = {
            'boundary': int(np.sum(self.end_conditions == 1)),
            'field_error': int(np.sum(self.end_conditions == 2)),
            'low_field': int(np.sum(self.end_conditions == 3)),
            'timeout': int(np.sum(self.end_conditions == 4)),
            'active': n_active,
        }
        return {
            'total_carriers': n_total,
            'active_carriers': n_active,
            'inactive_carriers': n_total - n_active,
            'average_steps': avg_steps,
            'max_steps': max_steps,
            'carrier_type': self.carrier_type,
            'end_conditions': end_condition_counts,
        }

    def get_performance_stats(self):
        return self.performance_stats.copy()

    def update_original_carriers(self, original_carriers):
        logger.info("更新 %s 状态", self.carrier_type)
        updated_count = 0
        for i, carrier in enumerate(original_carriers):
            if i >= len(self.positions):
                break
            try:
                carrier.x = float(self.positions[i][0])
                carrier.y = float(self.positions[i][1])
                carrier.z = float(self.positions[i][2])
                carrier.t = float(self.times[i])
                x_reduced, y_reduced = self.reduced_positions[i]
                carrier.x_reduced = float(x_reduced)
                carrier.y_reduced = float(y_reduced)
                x_num, y_num = self._calculate_electrode_numbers(carrier.x, carrier.y)
                carrier.x_num = x_num
                carrier.y_num = y_num
                carrier.path = [[float(p[0]), float(p[1]), float(p[2]), float(p[3])] for p in self.paths[i]]
                carrier.path_reduced = [
                    [float(p[0]), float(p[1]), float(p[2]), float(p[3]), int(p[4]), int(p[5])]
                    for p in self.paths_reduced[i]
                ]
                self._reinitialize_signal_list(carrier)
                if not self.active[i]:
                    condition_map = {1: "超出边界", 2: "电场错误", 3: "低电场", 4: "超时"}
                    carrier.end_condition = condition_map.get(self.end_conditions[i], "未知")
                updated_count += 1
            except Exception as exc:  # pragma: no cover - 容错逻辑
                logger.warning("更新载流子 %d 失败: %s", i, exc)
        logger.info("已更新 %d 个载流子", updated_count)
        return updated_count

    def _reinitialize_signal_list(self, carrier):
        try:
            if getattr(carrier, 'read_out_contact', None):
                if len(carrier.read_out_contact) == 1:
                    x_span = carrier.read_out_contact[0].get('x_span', 0)
                    y_span = carrier.read_out_contact[0].get('y_span', 0)
                    carrier.signal = [[] for _ in range((2 * x_span + 1) * (2 * y_span + 1))]
                else:
                    carrier.signal = [[] for _ in range(len(carrier.read_out_contact))]
            else:
                carrier.signal = [[]]
        except Exception:  # pragma: no cover - 容错逻辑
            carrier.signal = [[]]


def generate_electron_images(electron_system, save_dir="electron_images"):
    """示例可视化辅助函数，保留与优化版一致的接口。"""
    if electron_system.carrier_type != "electron":
        logger.error("错误：提供的载流子系统不是电子类型")
        return
    if plt is None:
        logger.warning("matplotlib 未安装，无法生成图像")
        return
    os.makedirs(save_dir, exist_ok=True)
    electron_system.generate_electron_images(save_dir)
    logger.info("电子图像生成完成: %s", save_dir)
