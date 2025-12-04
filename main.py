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
import ppigrf # can calculate Earth's magnetic field for the magnetic torque function.
print("hi")
## Constants of the satellite
m = 20.1 # mass (kg)
cg = 0.01 * np.array([-0.51, 1.35, 2.02]) # center of gravity from geometric center (m)
inertia = np.diag([0.540, 0.518, 0.357]) # moment of inertia tensor (kg*m^2)
r_altitude = 400e3 # altitude (m)
r_earth = 6370e3 # radius of the Earth (m)
r_total = r_earth + r_altitude # total radius from center of the Earth (m)
mu_earth = 3.986e14 # gravitational parameter of the Earth (m^3/s^2)
raan = np.deg2rad(45) # right ascension of ascending node (degrees to radians)
inclination = np.deg2rad(51.6) # inclination of orbit (degrees to radians)
dimensions = 0.01 * np.array([67.89, 22.63, 32.65]) # dimensions of the satellite (m)
dipole_moment = np.array([0.2, 0.2, 0.2]) # magnetic dipole moment (A*m^2)


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
    n = np.sqrt(w**2 + x**2 + y**2 + z**2) # normalize quaternion to avoid numerical integration drift
    w, x, y, z = w/n, x/n, y/n, z/n
    C = np.array([
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
    omega_M = np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])
    return omega_M
## External Torques

torque_history = { 
    "time": [],
    "gravity_gradient": [],
    "magnetic": [],
    "aero": [],
    "srp": [],
    "total": []
} # for data collection and plotting

def gravity_gradient_torque(mu_earth, r_i, q, inertia): # r_i is position in inertial frame
    rnorm = np.linalg.norm(r_i)
    rhat_i = r_i / rnorm

    rhat_b = inertial_to_body(rhat_i, q) # convert inertial position to body frame position
    T_gravity = 3 * mu_earth * np.cross(rhat_b, (inertia @ rhat_b)) / rnorm**3
    return T_gravity

def magnetic_torque(dipole, magfield_i, q): # magnetic field is inputted in inertial frame, will be converted
    magfield_b = inertial_to_body(magfield_i, q) 
    T_magnetic = np.cross(dipole,magfield_b)
    return T_magnetic

def drag_torque():
    T_drag = np.array([0, 0, 0]) # not sure how to find drag yet, specifcally the position vector from COM to center of pressure
    return T_drag

def srp_torque():
    T_srp = np.array([0, 0, 0]) # not sure how to find solar radiation pressure torque yet, same problem as drag torque
    return T_srp


## dynamics simualation functions

def orbit_r_v_calculation(t): # inputs a time, and outputs the position and velocity vectors in inertial frame
    n_mean = np.sqrt(mu_earth / r_total**3) # mean motion (radians/s)
    theta = n_mean * t # mean anomaly (radians)
    r_orbit = r_total * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])
    r_inclination = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination), np.cos(inclination)]
    ])
    r_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])
    r_i = r_raan @ r_inclination @ r_orbit

    v_mag = np.sqrt(mu_earth / r_total)
    v_orbit = v_mag * np.array([
        -np.sin(theta),
        np.cos(theta),
        0
    ])
    v_inclination = r_inclination
    v_raan = r_raan
    v_i = v_raan @ v_inclination @ v_orbit
    return r_i, v_i

def magnetic_field(r_i): # Simplfied magnetic field: https://en.wikipedia.org/wiki/Magnetic_dipole
    rnorm = np.linalg.norm(r_i)
    rhat = r_i / rnorm
    mu0 = 4 * np.pi * 1e-7
    M_earth = 7.94e22
    dipole_tilt = np.deg2rad(11)
    mhat = np.array([
        np.sin(dipole_tilt),
        0,
        np.cos(dipole_tilt)
    ])

    B = (mu0 * M_earth/ (4 * np.pi * rnorm**3)) * ((3 * (mhat.dot(rhat))*rhat) - mhat)
    return B

def state_vector_equation(t, y):
    q = y[0:4]
    omega = y[4:7]
    q = quaternion_norm(q)

    r_i, v_i = orbit_r_v_calculation(t)

    torque_mag = magnetic_torque(dipole_moment, magnetic_field(r_i), q)
    torque_grav = gravity_gradient_torque(mu_earth, r_i, q, inertia)
    torque_drag = drag_torque() # need to finish
    torque_srp = srp_torque() # need to finish

    torque_total = torque_mag + torque_grav + torque_drag + torque_srp

    torque_history["time"].append(t)
    torque_history["aero"].append(torque_drag.copy())
    torque_history["gravity_gradient"].append(torque_grav.copy())
    torque_history["magnetic"].append(torque_mag.copy())
    torque_history["srp"].append(torque_srp.copy())
    torque_history["total"].append(torque_total.copy())

    omega_mat = omega_matrix(omega)

    q_dot = 0.5 * omega_mat.dot(q)
    omega_dot = np.linalg.inv(inertia) @ (torque_total - np.cross(omega, (inertia @ omega)))

    dydt = np.zeros(7) # create the change in state vector
    dydt[0:4] = q_dot
    dydt[4:7] = omega_dot
    return dydt

q_initial = np.array([1, 0, 0, 0])
omega_initial = np.array([0, 0, 0.01])

state_initial = np.concatenate([q_initial, omega_initial])

def rk4_integrator(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2.0, y + dt/2.0 * k1)
    k3 = func(t + dt/2.0, y + dt/2.0 * k2)
    k4 = func(t + dt,     y + dt * k3)

    y_next = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    q_next = quaternion_norm(y_next[0:4])
    y_next[0:4] = q_next
    return y_next

## Example Values
t_start = 0.0
t_end = 6000.0
dt = 0.1
y = state_initial.copy()

t = t_start
history = []
times = []
pos_history = []
vel_history = []

while t < t_end:
    y = rk4_integrator(state_vector_equation, t, y, dt)
    t += dt
    history.append(y.copy())
    times.append(t)

    # Save position and velocity
    r_i, v_i = orbit_r_v_calculation(t)
    pos_history.append(r_i.copy())
    vel_history.append(v_i.copy())

history = np.array(history)
times = np.array(times)
pos_history = np.array(pos_history)
vel_history = np.array(vel_history)

for key in ["gravity_gradient", "magnetic", "aero", "srp", "total"]:
    torque_history[key] = np.array(torque_history[key])
torque_history["time"] = np.array(torque_history["time"])

# plotting results
import matplotlib.pyplot as plt
time = torque_history["time"]
# Gravity Gradient Torque components
plt.figure()
plt.plot(time, torque_history["gravity_gradient"][:,0], label="Gravity X")
plt.plot(time, torque_history["gravity_gradient"][:,1], label="Gravity Y")
plt.plot(time, torque_history["gravity_gradient"][:,2], label="Gravity Z")
plt.title("Gravity Gradient Torque Components")
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid(True)

# Magnetic Torque components
plt.figure()
plt.plot(time, torque_history["magnetic"][:,0], label="Magnetic X")
plt.plot(time, torque_history["magnetic"][:,1], label="Magnetic Y")
plt.plot(time, torque_history["magnetic"][:,2], label="Magnetic Z")
plt.title("Magnetic Torque Components")
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid(True)

# Position Components
plt.figure()
plt.plot(times, pos_history[:,0], label="Position X")
plt.plot(times, pos_history[:,1], label="Position Y")
plt.plot(times, pos_history[:,2], label="Position Z")
plt.title("Position Components")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)

# Velocity Components
plt.figure()
plt.plot(times, vel_history[:,0], label="Velocity X")
plt.plot(times, vel_history[:,1], label="Velocity Y")
plt.plot(times, vel_history[:,2], label="Velocity Z")
plt.title("Velocity Components")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid(True)
plt.show()