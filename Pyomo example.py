com_position = [0, 0, 0.32] 
com_velocity = [0, 0, 0]
com_roll_pitch_yaw = [0., -0., 0.]
com_angular_velocity = [0, 0, 0]
foot_contact_states = [1, 1, 1, 1]
foot_positions_base_frame = [0.23384717, -0.12642096, -0.25028953, 0.23002377, 0.13765568, -0.25137764, -0.11875363, -0.13255282, -0.2428782, -0.12328772, 0.13151313, -0.24395156]
foot_friction_coeffs = [0.45, 0.45, 0.45, 0.45]
desired_com_position = [0., 0., 0.32]
desired_com_velocity = [0., 0., 0.]
desired_com_roll_pitch_yaw = [0., 0., 0.]
desired_com_angular_velocity = [0., 0., 0.]  
body_inertia = np.array([0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447])
body_mass = 108/9.8
g = -9.8
Ts = 0.025
mu = 0.45

from scipy.spatial.transform import Rotation as R
def standing_mpc_controller(
        com_position,
        com_velocity,
        com_roll_pitch_yaw,
        com_angular_velocity,
        foot_contact_states,
        foot_positions_base_frame,
        foot_friction_coeffs,
        desired_com_position,
        desired_com_velocity,
        desired_com_roll_pitch_yaw,
        desired_com_angular_velocity,
):
    '''
        [0],  #com_position
        np.asarray(self._state_estimator.com_velocity_body_frame,
                dtype=np.float64),  #com_velocity
        np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
        # Angular velocity in the yaw aligned world frame is actually different
        # from rpy rate. We use it here as a simple approximation.
        np.asarray(self._robot.GetBaseRollPitchYawRate(),
                dtype=np.float64),  #com_angular_velocity
        foot_contact_state,  #foot_contact_states
        np.array(self._robot.GetFootPositionsInBaseFrame().flatten(),
                dtype=np.float64),  #foot_positions_base_frame
        self._friction_coeffs,  #foot_friction_coeffs
        desired_com_position,  #desired_com_position
        desired_com_velocity,  #desired_com_velocity
        desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
        desired_com_angular_velocity  #desired_com_angular_velocity
        A, B, Q, R, N, x0, uL, uU
    '''

    ### parameters for models

    # Euler angle xyz to zyx conversion use scipy
    Rz = np.array([[np.cos(com_roll_pitch_yaw[2]), np.sin(com_roll_pitch_yaw[2]), 0],
                   [-np.sin(com_roll_pitch_yaw[2]), np.cos(com_roll_pitch_yaw[2]), 0],
                   [0, 0, 1]])

    zeros = np.zeros((3, 3))
    ones = np.eye(3)
    A = np.block([[ones, zeros, Rz * Ts, zeros],
                  [zeros, ones, zeros, ones * Ts],
                  [zeros, zeros, ones, zeros],
                  [zeros, zeros, zeros, ones]])
    GI = Ts * np.linalg.inv(body_inertia.reshape((3, 3)))
    B4 = (Ts / body_mass) * ones

    # [r1] is a skew symmetric matrix for cross product
    def SkewThis(vec):
        skewedup = np.array([[0, -vec[2], vec[1]],
                             [vec[2], 0, -vec[0]],
                             [-vec[1], vec[0], 0]])
        return skewedup

    B = np.block([[zeros, zeros, zeros, zeros],
                  [zeros, zeros, zeros, zeros],
                  [GI @ SkewThis(foot_positions_base_frame[0:3]), GI @ SkewThis(foot_positions_base_frame[3:6]),
                   GI @ SkewThis(foot_positions_base_frame[6:9]), GI @ SkewThis(foot_positions_base_frame[9:12])],
                  [B4, B4, B4, B4]])
    model = pyo.ConcreteModel()
    model.N = 10
    model.nx = 12
    model.nu = 12

    model.Q = np.diag([1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1])
    model.R = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    g_hat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g * Ts]).T

    x_desired = np.hstack([desired_com_roll_pitch_yaw,
                           desired_com_position,
                           desired_com_angular_velocity,
                           desired_com_velocity])
    x0 = np.hstack([com_roll_pitch_yaw, com_position, com_angular_velocity, com_velocity])
    model.tIDX = pyo.Set(initialize=range(0, model.N + 1))
    model.xIDX = pyo.Set(initialize=range(0, model.nx))
    model.uIDX = pyo.Set(initialize=range(0, model.nu))

    # these are 12x12 arrays:
    model.A = A
    model.B = B

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)

    # Objective:
    '''
    model.cost = pyo.Objective(
        expr=(sum(Q[i] * (model.x[i, t] - x_desired[i]) ** 2 for t in model.tIDX for i in model.xIDX) + sum(
            R[j] * (model.u[j, t]) ** 2 for t in model.tIDX for j in model.uIDX)), sense=pyo.minimize)
'''

    def objective_rule(model):
        costX = 0.0
        costU = 0.0
        for t in model.tIDX:
            for i in model.xIDX:
                for j in model.xIDX:
                    if t < model.N:
                        costX += (model.x[i, t] - x_desired[i]) * model.Q[i,j] * (model.x[j, t] - x_desired[j])
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i,j] * model.u[j, t]
        return costX + costU

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    # Constraints:
    # double check
    def equality_const_rule(model, i, t):
        return model.x[i, t + 1] - (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
                                    + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) + g_hat[
                                        i]) == 0.0 if t < model.N else pyo.Constraint.Skip

    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=equality_const_rule)
    # input constraints
    model.constraint1 = pyo.Constraint(model.xIDX, rule=lambda model, i: model.x[i, 0] == x0[i])

    # constraint on -fz > 0 for 4 feet
    model.constraint2 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[2, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint3 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[5, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint4 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[8, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint5 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[11, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    # constraint |fx| <= -mu*fz
    model.constraint6 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[0, t] - mu * model.u[2, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint7 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[3, t] - mu * model.u[5, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint8 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[6, t] - mu * model.u[8, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint9 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[9, t] - mu * model.u[11, t] <= 0
    if t < model.N else pyo.Constraint.Skip)

    model.constraint10 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[0, t] - mu * model.u[2, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint11 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[3, t] - mu * model.u[5, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint12 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[6, t] - mu * model.u[8, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint13 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[9, t] - mu * model.u[11, t] <= 0
    if t < model.N else pyo.Constraint.Skip)

    # constraint |fy| <= -mu*fz
    model.constraint14 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[1, t] - mu * model.u[2, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint15 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[4, t] - mu * model.u[5, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint16 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[7, t] - mu * model.u[8, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint17 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[10, t] - mu * model.u[11, t] <= 0
    if t < model.N else pyo.Constraint.Skip)

    model.constraint18 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[1, t] - mu * model.u[2, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint19 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[4, t] - mu * model.u[5, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint20 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[7, t] - mu * model.u[8, t] <= 0
    if t < model.N else pyo.Constraint.Skip)
    model.constraint21 = pyo.Constraint(model.tIDX, rule=lambda model, t: -model.u[10, t] - mu * model.u[11, t] <= 0
    if t < model.N else pyo.Constraint.Skip)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)
  
    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False
            
    xOpt = np.asarray([[model.x[i,t]() for t in model.tIDX] for i in model.xIDX])
    uOpt = np.asarray([[model.u[j,t]() for t in model.tIDX] for j in model.uIDX])
    force = np.array(uOpt)
    # force = np.ndarray.flatten(uOpt)

    JOpt = model.cost()
    return force, xOpt, JOpt, A, B, g_hat
[force, xOpt, JOpt, A, B, g_hat] = standing_mpc_controller(com_position, com_velocity, com_roll_pitch_yaw, com_angular_velocity,
                        foot_contact_states, foot_positions_base_frame, foot_friction_coeffs,
                        desired_com_position, desired_com_velocity, desired_com_roll_pitch_yaw,
                        desired_com_angular_velocity)

print(xOpt.shape)
print(force.shape)
for k in range(10-1):
  print(xOpt[:, k+1] - A@xOpt[:, k] - B@force[:, k] - g_hat)
print(force)
print(xOpt)
print(JOpt)
plt.plot(force[2, :])
plt.plot(force[5, :])
plt.plot(force[8, :])
plt.plot(force[11, :])
