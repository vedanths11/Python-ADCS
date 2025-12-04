"""
Satellite Attitude Dynamics Equations
https://arxiv.org/pdf/2408.00176
    Euler's equation for a rigid body:
I * omegadot = Total Torque - omega x (I * omega) 
gyroscopic effects within the satellite could not be quantified

    External Torques: https://bibliotekanauki.pl/articles/277322
1. Gravity Gradient Torque - differences in forces on different parts of the satellite, gravity will have slightly different forces
causing a moment around the satellite.
2. Magnetic Field Interaction - due to the Earth's magnetic field that interacts with a dipole moment of the satellite.
    Torque(magnetic field) = satellite's magnetic moment x Earth's agnetic field
    µ= nIA, where n is the number of turns in the magnetorquer solenoid, I is the current, and A is the vector area of the solenoid.
3. Aerodynamic Drag - air density (atmosphere) in LEO results in a drag force acting on the satellite.
4. Environmental Disturbances (SRP) - pressure is generated on the satellite when the sun's photons are reflected and absorbed.

    Expanded Equations of Motion
I(tot) * omegadot(body) = T(extertnal) - omega(body) x (I(tot) * omega(body)) - T(wheels) - omega(body) x EwS(angular momentum wheels)
qdot = 0.5 * q * omega(body)
T(wheels) = f(q(err), g(err), w(err). T(wheels) is the control torque from the reaction wheels

These equations include the internal torques from the reaction wheels and gyroscopic effects from the wheels for higher precision.
    
"""
import numpy as np
import sympy as sp

## Constants of the satellite
m = 20.1 # mass (kg)
cg = sp.Matrix([-0.51, 1.25, 2.02]) # center of gravity from geometric center (cm)
inertia = sp.diag(0.540, 0.518, 0.357) # moment of inertia tensor (kg*m^2)
r_altitude = 400e3 # altitude (m)
r_earth = 6370e3 # radius of the Earth (m)
r_total = r_earth + r_altitude # total radius from center of the Earth (m)
mu_earth = 3.986e14 # gravitational parameter of the Earth (m^3/s^2)
raan = sp.rad(45) # right ascension of ascending node (degrees to radians)
inclination = sp.rad(51.6) # inclination of orbit (degrees to radians)
dimensions = sp.Matrix([67.89, 22.63, 32.65]) # dimensions of the satellite (cm)
dipole_moment = sp.Matrix([0.2, 0.2, 0.2]) # magnetic dipole moment (A*m^2)

## Quaternion math functions
def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1                                                 # quaternion is defined as w (real) and x,y,z (complex) values
    w2, x2, y2, z2 = q2
    return np.array([                                                   # hamilton product of quaternions
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_norm(q):
    return q/np.linalg.norm(q)

def quaternion_inv(q):
    return quaternion_conjugate(q)/((np.linalg.norm(q))**2)

def quaternion_to_DCM(q): # Direction Cosine Matrix from quaternion that transforms from interial frame to body frame
    w, x, y, z = q
    n = sp.sqrt(w**2 + x**2 + y**2 + z**2) # normalize quaternion to avoid numerical integration drift
    w, x, y, z = w/n, x/n, y/n, z/n
    C = sp.Matrix([
        [w**2 + x**2 - y**2 - z**2, 2*(x*y + w*z)            , 2*(x*z - w*y)],
        [2*(x*y - w*z)            , w**2 - x**2 + y**2 - z**2, 2*(y*z + w*x)],
        [2*(x*z + w*y)            , 2*(y*z - w*x)            , w**2 - x**2 - y**2 + z**2]
    ])
    return C

def inertial_to_body(vec_i, q):
    C_ib = quaternion_to_DCM(q)
    vec_b = C_ib @ vec_i
    return vec_b

def body_to_inertial(vec_b, q):
    C_ib = quaternion_to_DCM(q)
    C_bi = C_ib.T
    vec_i = C_bi @ vec_b
    return vec_i

def omega_matrix(omega):
    wx, wy, wz = omega
    omega_M = sp.Matrix([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])
    return omega_M
## External Torques

def gravity_gradient_torque(mu_earth, r_i, q, inertia): # r_i is position in inertial frame
    rnorm = sp.norm(r_i)
    rhat_i = r_i / rnorm

    rhat_b = inertial_to_body(rhat_i, q) # convert inertial position to body frame position
    T_gravity = 3 * mu_earth * rhat_b.cross(inertia @ rhat_b) / rnorm**3
    return T_gravity

def magnetic_torque(dipole, magfield_i, q): # magnetic field is inputted in inertial frame, will be converted
    magfield_b = inertial_to_body(magfield_i, q) 
    T_magnetic = dipole.cross(magfield_b)
    return T_magnetic

def drag_torque():
    T_drag = sp.Matrix([0, 0, 0]) # not sure how to find drag yet, specifcally the position vector from COM to center of pressure
    return T_drag

def srp_torque():
    T_srp = sp.Matrix([0, 0, 0]) # not sure how to find solar radiation pressure torque yet, same problem as drag torque
    return T_srp


## dynamics simualation functions

def orbit_r_v_calculation(t): # inputs a time, and outputs the position and velocity vectors in inertial frame
    n_mean = sp.sqrt(mu_earth / r_total**3) # mean motion (radians/s)
    theta = n_mean * t # mean anomaly (radians)
    r_orbit = r_total * sp.Matrix([
        sp.cos(theta),
        sp.sin(theta),
        0
    ])
    r_inclination = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(inclination), -sp.sin(inclination)],
        [0, sp.sin(inclination), sp.cos(inclination)]
    ])
    r_raan = sp.Matrix([
        [sp.cos(raan), -sp.sin(raan), 0],
        [sp.sin(raan), sp.cos(raan), 0],
        [0, 0, 1]
    ])
    r_i = r_raan @ r_inclination @ r_orbit

    v_mag = sp.sqrt(mu_earth / r_total)
    v_orbit = v_mag * sp.Matrix([
        -sp.sin(theta),
        sp.cos(theta),
        0
    ])
    v_inclination = r_inclination
    v_raan = r_raan
    v_i = v_raan @ v_inclination @ v_orbit
    return r_i, v_i