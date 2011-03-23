# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:28:29 2011

@author: ba1224

Generates data for a metal fermi surface - taken from Schoenberg


"""


import numpy as np
import sys


K_BOLTZ = 1.3806503e-23
E_CHARGE = 1.60217646e-19
HBAR = 1.05457148e-34
E_MASS = 9.10938188e-31


def a_torque(b, theta):
    theta = np.deg2rad(theta)
    return b * np.abs(np.sin(theta))

def a_temp(b, m_therm=1, t=sys.float_info.min*1e3):
    # Precalculate the large constant so that the floats will not underflow
    C = 2 * np.pi**2 * K_BOLTZ * E_MASS / (E_CHARGE * HBAR)
    X = C * m_therm * t / (b)
    return X / np.sinh(X)

def a_dingle(b, f, l0=sys.float_info.max*1e-3):
    return np.exp(-np.pi * np.sqrt(2 * HBAR * f / E_CHARGE) / (l0 * b))

def a_mos(b, theta, f, d_theta=0):
#    theta = np.deg2rad(theta)
#    return np.exp(-(np.pi * f * np.sin(theta) * d_theta / (b * np.cos(theta))) ** 2)
    return 1
    
def a_dop(b, theta, a, d_p=0):
#    theta = np.deg2rad(theta)
#    return np.exp(-(np.pi**2 * HBAR * d_p / (a**2 * E_MASS * b * np.cos(theta)))) 
    return 1
    
def a_spin(m_sus=0):
    g = 2
    return np.abs(np.cos(np.pi * g * m_sus / (2 * E_MASS)))   
    
def a_warp(b, theta, phi):
    # TODO
    theta = np.deg2rad(theta)
    return 1


def sim_amplitude(b, f, theta=45, phi=0, t=0.0, m_therm=1, l0=sys.float_info.max*1e-3, m_sus=0, a=5, d_p=0, d_theta=0, a0=1):
    '''
    Find the amplitude of the oscillations based on the NJPhys review article on Tl2201 by Pat
    This features approximations that are only suitable for quasi 2d fermi-surfaces

    Call with:
        sim_amplitude(b, f, theta=0, phi=0, t=0.0, l0=sys.float_info.max * 1e-3, m_sus=0, a=5, d_p=0, d_theta=0, a0=1)
        
    Args:
        b = field value(s)
        f = frequency of oscillation
        theta = angle from azimuth
        phi = angle of rotation
        t = temperature
        l0 = mean free path
        m_sus = Susceptibility mass
        a = a lattice parameter
        d_p = variation in hole doping across sample
        d_theta = variation in alignment of subdomains of crystal
        a0 = an amplification factor (to e.g. scale torque to a voltage)
        
    Notes:
        Defaults chosen so that various terms will equal 1 if not specified
        Warping term not yet implemented, phi therefore does nothing at present
    '''
    a = a0 * a_torque(b, theta) * a_temp(b, m_therm, t) * a_dingle(b, f, l0) \
        * a_mos(b, theta, f, d_theta) * a_dop(b, theta, a, d_p) * a_spin(m_sus) \
        * a_warp(b, theta, phi)
    return a
    
def sim_oscillation(b, f, theta=45, phi=0, t=0.0, m_therm=1, l0=sys.float_info.max*1e-3, m_sus=0, a=5, d_p=0, d_theta=0, a0=1):
    '''
    Returns the oscillations in inverse field as would eb measured on a cantilever 
    for example. Based on the NJPhys review article on Tl2201 by Pat
    This features approximations that are only suitable for quasi 2d fermi-surfaces

    Call with:
        sim_oscillation(b, f, theta=0, phi=0, t=0.0, l0=sys.float_info.max * 1e-3, m_sus=0, a=5, d_p=0, d_theta=0, a0=1)
        
    Args:
        b = field value(s)
        f = frequency of oscillation
        theta = angle from azimuth
        phi = angle of rotation
        t = temperature
        l0 = mean free path
        m_sus = Susceptibility mass
        a = a lattice parameter
        d_p = variation in hole doping across sample
        d_theta = variation in alignment of subdomains of crystal
        a0 = an amplification factor (to e.g. scale torque to a voltage)
        
    Notes:
        Defaults chosen so that various terms will equal 1 if not specified
        Warping term not yet implemented, phi therefore does nothing at present
    '''
    o = np.sin(2 * np.pi * f / b) * sim_amplitude(b, f, theta, phi, t, m_therm, l0, m_sus, a, d_p, d_theta, a0)
    return o
    
# Some tests
if __name__ == '__main__':
    from pylab import *
    # All the following should be unity (or very close to it) by default    
    b = np.linspace(10, 18, 25)        
    theta = 90
    f = 2500
    # Dingle term with 'infinite' mean free path
#    amp = a_dingle(b, f) # Default is l0 = sys.float_info.max * 1e-3
    # Temperature term at t=0
#    amp = a_temp(b, theta) # Default is t = sys.float_info.min * 1e3
    # Torque at theta=90
#    amp = a_torque(b, theta)
    # Variation in doping 
#    amp = a_dop(b, theta, a=12) # Default d_p = 0
    # Variations in mosaicity
#    amp = a_mos(b, theta, f)    # Default d_theta = 0
    # Spin susceptibility term
    amp = a_spin() * np.ones_like(b)    
    plot(b, amp)
    show()



