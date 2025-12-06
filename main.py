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
    

    Controls Equations: Section 6 of https://arxiv.org/pdf/2408.00176
E(wheels) = 0.5 * m(wheels) * r(wheels)^2 * [e1 e2 e3] for each direction e the wheel faces: E is like an inertia tensor vector for wheels.
torque(wheels) = -kp * q(err) - ki * g(err) - kd * w(err), simplfies when split into diagonal matrix form (see pg.21 of first link)
g(err) is the time weighted sum of q(err), g(err) (t) = integral from 0 to t of q(err) (t') * e ^ (t' - t / t0) dt'
I(tot) * Wdot(b) = -E(wheels) * sdot + torque(ext) - w(b) x (I(tot) @ w(b)) - w(b) x (E(wheels) @ s)
wmax(wheels) = 0.5 * m(wheels) * r(wheels)^2 / I(min) - approximates the wheels angular velocity

    Gain Determination
Use a set of differential equations to derive optimal gains instead of manual tunings.
Approximate the system as a linear system about equillibrium and use te Jacobian Matrix

"""
import numpy as np
import ppigrf # can calculate Earth's magnetic field for the magnetic torque function.

## Constants of the satellite
m = 20.1 # mass (kg)
cg = 0.01 * np.array([-0.51, 1.35, 2.02]) # center of gravity from geometric center (m)
inertia = np.diag([0.540, 0.518, 0.357]) # moment of inertia tensor (kg*m^2)
r_altitude = 400e3 # altitude (m)
r_earth = 6371e3 # radius of the Earth (m)
r_total = r_earth + r_altitude # total radius from center of the Earth (m)
mu_earth = 3.986e14 # gravitational parameter of the Earth (m^3/s^2)
raan = np.deg2rad(45) # right ascension of ascending node (degrees to radians)
inclination = np.deg2rad(51.6) # inclination of orbit (degrees to radians)
dimensions = 0.01 * np.array([67.89, 22.63, 32.65]) # dimensions of the satellite (m)
dipole_moment = np.array([0.2, 0.2, 0.2]) # magnetic dipole moment (A*m^2)

## Control Law Constants
dt_control = 0.1 # time step (s)
t0_integ = 10 # scaling factor (s) - how much time it takes for the controller to forget past errors (i.e errors from 10 seconds ago are forgotten)
g_err = np.array([0.0, 0.0, 0.0]) # time weighted quaternion error
q_target = np.array([1, 0, 0, 0]) # target quaternion (can be changed based on pointing requirements)
rho = 0.01 # gain scale, the paper used 0.05

## from page 26 of https://arxiv.org/pdf/2408.00176, gain matrices are found
Ix, Iy, Iz = inertia[0,0], inertia[1,1], inertia[2,2]
kd = np.array([
    rho * ((3 * Ix / dt_control) - (Ix/t0_integ)),
    rho * ((3 * Iy / dt_control) - (Iy/t0_integ)),
    rho * ((3 * Iz / dt_control) - (Iz/t0_integ))
])

kp = np.array([
    2 * ((Ix**2) - Ix*kd[0]*t0_integ + (kd[0]**2)*(t0_integ**2)) / (3 * Ix * t0_integ**2),
    2 * ((Iy**2) - Iy*kd[1]*t0_integ + (kd[1]**2)*(t0_integ**2)) / (3 * Iy * t0_integ**2),
    2 * ((Iz**2) - Iz*kd[2]*t0_integ + (kd[2]**2)*(t0_integ**2)) / (3 * Iz * t0_integ**2)
])

ki = np.array([
    (2 * (kd[0]*t0_integ - 2*Ix)**3) / (27 * (Ix**2) * (t0_integ**3)),
    (2 * (kd[1]*t0_integ - 2*Iy)**3) / (27 * (Iy**2) * (t0_integ**3)),
    (2 * (kd[2]*t0_integ - 2*Iz)**3) / (27 * (Iz**2) * (t0_integ**3))
])


## Reaction Wheel Constants
m_wheel = 0.2 # mass (kg)
r_wheel = 0.05 # radius (m)
I_wheel = 0.5 * m_wheel * (r_wheel**2) # inertia (kg * m^2)
E_wheel = I_wheel * np.eye(3) # inertia matrix (3x3 diagonal matrix)
s_wheel = np.array([0.0, 0.0, 0.0]) # initial wheel speed (rad/s)
smax_wheel = 3000 * (2 * np.pi/60) # max wheel speed (3000 rpm to rad/s)
omega_max_x = (I_wheel/Ix) * smax_wheel
omega_max_y = (I_wheel/Iy) * smax_wheel
omega_max_z = (I_wheel/Iz) * smax_wheel
omega_max_vec = np.array([omega_max_x, omega_max_y, omega_max_z])

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
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

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

## Control Law Functions

def quaternion_error(q_current, q_target):
    # q_err = q_target * q_current^-1
    q_err = quaternion_mult(q_target, quaternion_inv(q_current))
    q_err = quaternion_norm(q_err)
    return q_err

def update_time_weighted_sum_error(q_err_vec, g_err, dt, t0):
    g_err_new = g_err + (dt * (q_err_vec - g_err) / t0)
    return g_err_new

def target_omega(q_err, omega_max_vec, dt_control):
    theta = 2 * np.arccos(np.clip(q_err[0], -1, 1)) # find the theta angle using the quaternion definition of half angles, makes sure that it is between -1 and 1
    half_theta = theta/2
    if abs(np.sin(half_theta)) < 1e-10:
        return np.zeros(3) # avoid dividing by 0 later
    n_hat = q_err[1:4] / np.sin(half_theta) # error axis

    phi = theta*n_hat
    omega_desired = phi / dt_control
    omega_target = np.clip(omega_desired, -omega_max_vec, omega_max_vec)
    return omega_target

def PID(q_current, omega_current, q_target, g_err, omega_max_vec):
    q_err = quaternion_error(q_current, q_target)
    q_err = quaternion_norm(q_err)
    q_err_vector = q_err[1:4]

    g_err_new = update_time_weighted_sum_error(q_err_vector, g_err, dt_control, t0_integ)

    omega_target = target_omega(q_err, omega_max_vec, dt_control)

    omega_err = omega_target - omega_current
    max_torque = 0.015 # Nm
    torque_wheels = np.clip((kp * q_err_vector) + (ki * g_err_new) + (kd*omega_err), -max_torque, max_torque)
    return torque_wheels, g_err_new
## External Torques

torque_history = { 
    "time": [],
    "gravity_gradient": [],
    "magnetic": [],
    "drag": [],
    "srp": [],
    "wheels": [],
    "pointing_error": [],
    "omega": [],
    "quaternion": [],
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

def state_vector_equation(t, y, torque_wheels, s_dot):
    q = y[0:4]
    omega = y[4:7]
    s = y[7:10]
    q = quaternion_norm(q)

    r_i, v_i = orbit_r_v_calculation(t)

    torque_mag = magnetic_torque(dipole_moment, magnetic_field(r_i), q)
    torque_grav = gravity_gradient_torque(mu_earth, r_i, q, inertia)
    torque_drag = drag_torque() # need to finish
    torque_srp = srp_torque() # need to finish

    torque_total = torque_mag + torque_grav + torque_drag + torque_srp
    
    I_tot = inertia + E_wheel

    omega_mat = omega_matrix(omega)
    q_dot = 0.5 * omega_mat.dot(q)

    body_torques = torque_total + torque_wheels - np.cross(omega, (I_tot @ omega)) - np.cross(omega, E_wheel @ s)
    omega_dot = np.linalg.solve(I_tot, body_torques)

    q_err = quaternion_error(q, q_target)
    pointing_error = 2 * np.arccos(np.clip(q_err[0], -1, 1)) * (180/np.pi)

    dydt = np.zeros(10) # create the change in state vector
    dydt[0:4] = q_dot
    dydt[4:7] = omega_dot
    dydt[7:10] = s_dot

    return dydt

def rk4_integrator(func, t, y, dt, torque_wheels, s_dot):
    y[0:4] = quaternion_norm(y[0:4])
    k1 = func(t, y, torque_wheels, s_dot)
    k2 = func(t + dt/2.0, y + dt/2.0 * k1, torque_wheels, s_dot)
    k3 = func(t + dt/2.0, y + dt/2.0 * k2, torque_wheels, s_dot)
    k4 = func(t + dt,     y + dt * k3, torque_wheels, s_dot)

    y_next = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    q_next = quaternion_norm(y_next[0:4])
    y_next[0:4] = q_next
    return y_next

## Example Values

q_initial = np.array([0.9, 0.1, 0.1, 0.1]) # random numbers for now
q_initial = quaternion_norm(q_initial)
omega_initial = np.array([0, 0, 0.01])
t_start = 0.0
t_end = 600.0
dt = 0.01
s_initial = np.array([0.0, 0.0, 0.0])
state_initial = np.concatenate([q_initial, omega_initial, s_initial])

y = state_initial.copy()

t = t_start
last_control = -dt_control
g_err_state = g_err.copy()

torque_wheels = np.zeros(3)
s_dot = np.zeros(3)

history = []
times = []
pos_history = []
vel_history = []

while t < t_end:
    
    if t - last_control >= dt_control - 1e-9:
        torque_wheels, g_err_state = PID(y[0:4], y[4:7], q_target, g_err_state, omega_max_vec)
        s_dot = np.linalg.solve(E_wheel, torque_wheels)
        last_control = t

        # Save position and velocity
        r_i, v_i = orbit_r_v_calculation(t)
        torque_mag = magnetic_torque(dipole_moment, magnetic_field(r_i), y[0:4])
        torque_grav = gravity_gradient_torque(mu_earth, r_i, y[0:4], inertia)

        q_err = quaternion_error(y[0:4], q_target)
        pointing_error = 2 * np.arccos(np.clip(q_err[0], -1, 1)) * (180 / np.pi)
        pos_history.append(r_i.copy())
        vel_history.append(v_i.copy())
        torque_history["time"].append(t)
        torque_history["gravity_gradient"].append(torque_grav.copy())
        torque_history["magnetic"].append(torque_mag.copy())
        torque_history["wheels"].append(torque_wheels.copy())
        torque_history["pointing_error"].append(pointing_error)
        torque_history["omega"].append(y[4:7].copy())
        torque_history["quaternion"].append(y[0:4].copy())
    y = rk4_integrator(state_vector_equation, t, y, dt, torque_wheels, s_dot)

    if np.any(np.isnan(y)) or np.any(np.abs(y[4:7]) > 10):
        print(f"Unstable at t = {t}")
        print(f"Omega: {y[4:7]}")
        print(f"Torque: {torque_wheels}")
        break

    t += dt
    history.append(y.copy())
    times.append(t)

for key in ["gravity_gradient", "magnetic", "drag", "srp", "total", "wheels", "omega", "quaternion"]:
    torque_history[key] = np.array(torque_history[key])
torque_history["time"] = np.array(torque_history["time"])
torque_history["pointing_error"] = np.array(torque_history["pointing_error"])


import matplotlib.pyplot as plt
# Pointing Error over time
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(torque_history["time"], torque_history["pointing_error"])
plt.title("Pointing Error over Time")
plt.xlabel("Time (s)")
plt.ylabel("Pointing Error (deg)")
plt.grid(True)

# Wheel Torque over time
plt.subplot(3,1,2)
plt.plot(torque_history["time"], torque_history["wheels"][:, 0], label = "Wheel X")
plt.plot(torque_history["time"], torque_history["wheels"][:, 1], label = "Wheel Y")
plt.plot(torque_history["time"], torque_history["wheels"][:, 2], label = "Wheel Z")
plt.title("Reaction Wheel Torques over Time")
plt.xlabel("Time (s)")
plt.ylabel("Reaction Wheel Torques (Nm)")
plt.legend()
plt.grid(True)

# Angular Velocity
plt.subplot(3,1,3)
plt.plot(torque_history["time"], torque_history["omega"][:, 0], label = "Omega X")
plt.plot(torque_history["time"], torque_history["omega"][:, 1], label = "Omega Y")
plt.plot(torque_history["time"], torque_history["omega"][:, 2], label = "Omega Z")
plt.title("Angular Velocity over time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid(True)




""""" Annotated for now, can change to see plots when needed.
# plotting torques
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
"""
print("hi")
plt.tight_layout()
plt.show()