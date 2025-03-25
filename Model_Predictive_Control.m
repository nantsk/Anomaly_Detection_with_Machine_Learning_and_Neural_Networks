% R-bar Mode with Model Predictive Control (MPC) and UKF
clc;
clear;
close all;

% Orbital parameters
alt = 450e3;            % Altitude of orbit (m)
Re = 6371e3;            % Earth radius (m)
mu = 3.986004418e14;    % Earth gravitational parameter (m^3/s^2)
r_orbit = Re + alt;     % Orbital radius (m)
n = sqrt(mu/r_orbit^3); % Mean motion (rad/s)

% Satellite parameters
m_target = 150;         % Target satellite mass (kg)
m_chaser = 350;         % Chaser satellite mass (kg)
thrust_x = 17;          % Total thrust (N)
num_thrusters = 4;      % Number of thrusters

% MPC parameters 
Ts = 0.5;               % Sampling time (s)
global Np Nc
Np = 150;                % Prediction horizon
Nc = 60;                % Control horizon
Q_base = diag([1e6, 1e6, 1e6, 5e6, 5e6, 5e6]);  % Position weights
R_base = diag([0.1, 0.1]);  % Control weights
S = 1e19 * eye(6);      % Terminal state weight

% Initialize Chebyshev filter parameters
filter_order = 4;
ripple_dB = 0.5;  % dB
cutoff_freq = 0.08;    % Hz
sample_freq = 1/Ts;

% Design the Chebyshev filter
[b_cheby, a_cheby] = cheby1(filter_order, ripple_dB, cutoff_freq/(sample_freq/2));

% Initialize filter states
filter_states_x = zeros(filter_order, 1);
filter_states_y = zeros(filter_order, 1);

% Lyapunov function weights
P = diag([1e18, 1e18, 1e18, 1e13, 1e13, 1e13]);
S_lyapunov = P;

% State and input constraints
u_max_scalar = ((thrust_x * num_thrusters) / m_chaser) * 0.5;  % Thrust limit
v_max = 2;            % Maximum velocity (m/s)
x_min = [-inf; -inf; -inf; -v_max; -v_max; -v_max];  % State constraints
x_max = [inf; inf; inf; v_max; v_max; v_max];
u_min = [-u_max_scalar; -u_max_scalar];     % Control input constraints
u_max = [u_max_scalar; u_max_scalar];

% Dead-band control parameters
dead_band_far = 0.005;       % Dead-band when far from target
dead_band_mid = 0.002;      % Dead-band for mid-range
dead_band_near = 0.0001;     % Dead-band when close to target 

% Pulse-width modulation parameters
pwm_period = 2;             % PWM period
pwm_min_duty = 0.2;         % Minimum duty cycle

% Coast phase parameters
coast_distance_threshold = 500; % Distance threshold for coasting (m)
coast_velocity_threshold = 1.5; % Velocity threshold for coasting (m/s)
coast_phase_active = false; % Flag to track if coast phase is active

% Initial conditions
x0 = [-2903.1; -2991.5; 0; 1.0; 0.5; 0];  % 2.9031 km behind, 2.9915 km below
t_final = 1800;         % Simulation time (s)
t = 0:Ts:t_final;
N = length(t);

% Initialize state and control vectors
x = zeros(6, N);
u = zeros(2, N-1);
u_raw = zeros(2, N-1);
x(:,1) = x0;

% Corrected: Calculate initial distance and velocity
initial_distance = norm(x0(1:3));
initial_velocity = norm(x0(4:6));

disp(['Initial Distance: ', num2str(initial_distance), ' m, Initial Velocity: ', num2str(initial_velocity), ' m/s']);

% Initialize thruster firing points
thruster_firing_points = [];

% UKF parameters
nx = 6;                 % State dimension
nu = 2;                 % Control dimension
alpha = 1e-3;    % Scaling parameter
beta = 2;        % Optimal for Gaussian distributions
kappa = 0;       % Secondary scaling parameter

% State and measurement noise covariance
Q = diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]); % Process noise covariance
R = diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]); % Measurement noise covariance
L = length(x0);  % State dimension
lambda = alpha^2 * (L + kappa) - L; % Sigma point scaling factor
gamma = sqrt(L + lambda); % Square root of scaling factor

% UKF state and covariance initialization
x_ukf = x0; % Initial state estimate
P_ukf = eye(L); % Initial estimate covariance

% Chebyshev Filter Functions
function [b, a] = design_chebyshev_filter(order, ripple, cutoff_freq, sampling_freq)
    % Design lowpass Chebyshev Type I filter
    normalized_cutoff = cutoff_freq / (sampling_freq / 2);
    [b, a] = cheby1(order, ripple, normalized_cutoff, 'low');
end

function [u_filtered, filter_states_x, filter_states_y] = apply_chebyshev_filter(u_raw, filter_states_x, filter_states_y, b, a)
    % Apply filter to each control dimension
    [u_filtered(1), filter_states_x] = filter(b, a, u_raw(1), filter_states_x);
    [u_filtered(2), filter_states_y] = filter(b, a, u_raw(2), filter_states_y);
end


% Continuous-time state-space matrices
A_c = [0        0    0       1       0       0;
       0        0    0       0       1       0;
       0        0    0       0       0       1;
       3*n^2    0    0       0       2*n     0;
       0        0    0      -2*n     0       0;
       0        0   -n^2     0       0       0];
B_c = [0 0;
       0 0;
       0 0;
       1 0;
       0 1;
       0 0];

% Discrete-time state-space matrices
A_d = expm(A_c * Ts);
num_steps = 100;
tau = linspace(0, Ts, num_steps);
B_d = zeros(size(B_c));

for i = 1:num_steps
    B_d = B_d + expm(A_c * tau(i)) * B_c * (Ts / num_steps);
end

A = A_d;
B = B_d;

% Apply dead-band and PWM for thruster control
function u_modified = enhanced_thruster_control(u_raw, pos_norm, vel_norm, pwm_counter, phase, u_max_scalar)
    u_modified = u_raw;
    
    % Phase-specific parameters
    if phase == 1  % Far approach
        dead_band = 0.05 * u_max_scalar;
        pwm_period = 8;
        pwm_min_duty = 0.25;
    elseif phase == 2  % Mid-range
        dead_band = 0.03 * u_max_scalar;
        pwm_period = 6;
        pwm_min_duty = 0.3;
    elseif phase == 3  % Close approach
        dead_band = 0.01 * u_max_scalar;
        pwm_period = 4;
        pwm_min_duty = 0.4;
    else  % Final docking
        dead_band = 0.005 * u_max_scalar;
        pwm_period = 3;
        pwm_min_duty = 0.5;
    end
    
    % Incorporate velocity for adaptive dead-band
    velocity_factor = min(1.0, vel_norm / max(0.1, pos_norm/100));
    adjusted_dead_band = dead_band * (1 + velocity_factor);
    
    % Apply dynamic dead-band
    for i = 1:length(u_modified)
        if abs(u_modified(i)) < adjusted_dead_band
            u_modified(i) = 0;
        end
    end
    
    % Ensure pwm_counter works with the current period
    pwm_counter_adjusted = mod(pwm_counter-1, pwm_period) + 1;
    
    % Enhanced PWM implementation
    for i = 1:length(u_modified)
        mag = abs(u_modified(i));
        
        if mag > 0 && mag < 0.5 * u_max_scalar
            % Progressive duty cycle based on magnitude
            duty_cycle = pwm_min_duty + (1-pwm_min_duty) * (mag/(0.5*u_max_scalar));
            
            if pwm_counter_adjusted <= round(pwm_period * duty_cycle)
                u_modified(i) = sign(u_modified(i)) * min(u_max_scalar, mag / duty_cycle);
            else
                u_modified(i) = 0;
            end
        end
    end
    
    % Minimum impulse bit control for final approach
    if phase == 4 && pos_norm < 5
        min_impulse = 0.02 * u_max_scalar;
        for i = 1:length(u_modified)
            if abs(u_modified(i)) > 0 && abs(u_modified(i)) < min_impulse
                u_modified(i) = sign(u_modified(i)) * min_impulse;
            end
        end
    end
end

function [Q_phase, R_phase, u_constraints, horizon_Np, horizon_Nc] = phase_specific_control(phase, pos_norm, vel_norm, Q_base, R_base, Np_base, Nc_base, u_max)
    % Default initialization
    Q_phase = Q_base;
    R_phase = R_base;
    u_constraints = u_max;
    horizon_Np = Np_base;
    horizon_Nc = Nc_base;
    
    % Phase 1: Far approach (d > 1000m)
    if phase == 1
        Q_phase(1:3,1:3) = Q_base(1:3,1:3) * 1.0;  % Lower position weight
        Q_phase(4:6,4:6) = Q_base(4:6,4:6) * 1.0;  % Lower velocity weight
        R_phase = R_base * 1.0;  % Higher control penalty (energy saving)
        u_constraints = u_max * 0.7;  % Limit thrust for efficiency
        horizon_Np = 50;  % Longer prediction horizon
        horizon_Nc = 25;  % Longer control horizon
    
    % Phase 2: Mid-range approach (100m < d ≤ 1000m)
    elseif phase == 2
        Q_phase(1:3,1:3) = Q_base(1:3,1:3) * 0.5;
        Q_phase(4:6,4:6) = Q_base(4:6,4:6) * 0.8;
        R_phase = R_base * 2.0;
        u_constraints = u_max * 0.8;
        horizon_Np = 40;
        horizon_Nc = 20;
    
    % Phase 3: Close approach (10m < d ≤ 100m)
    elseif phase == 3
        Q_phase(1:3,1:3) = Q_base(1:3,1:3) * 10;  % Higher position weight
        Q_phase(4:6,4:6) = Q_base(4:6,4:6) * 5;
        R_phase = R_base * 0.1;  % Lower control penalty
        u_constraints = u_max * 0.9;
        horizon_Np = 30;
        horizon_Nc = 15;
    
    % Phase 4: Final docking (d ≤ 10m)
    else
        Q_phase(1:3,1:3) = Q_base(1:3,1:3) * 50.0;  % Much higher position weight
        Q_phase(4:6,4:6) = Q_base(4:6,4:6) * 5.0;   % Higher velocity weight
        R_phase = R_base * 0.05;   % Much lower control penalty
        u_constraints = u_max * 0.5;  % Finer control authority
        horizon_Np = 20;  % Shorter horizon for more reactive control
        horizon_Nc = 10;
    end
    
    % Velocity-specific adjustments
    if vel_norm > 0.5 * sqrt(pos_norm/100)
        % If velocity is high relative to distance, increase velocity weights
        Q_phase(4:6,4:6) = Q_phase(4:6,4:6) * 2.0;
    end
    
    % Final approach precision
    if phase == 4 && pos_norm < 1.0
        Q_phase(1:3,1:3) = Q_phase(1:3,1:3) * 3.0;  % Further increase position weights
        Q_phase(4:6,4:6) = Q_phase(4:6,4:6) * 2.0;  % Further increase velocity weights
    end
end

% Safety check function 
function [safe, safety_control] = check_safety(x, v_max, u_max_scalar)
    pos = x(1:3);
    vel = x(4:6);
    pos_norm = norm(pos);
    speed = norm(vel);
    safety_control = zeros(2,1);

        % Define dynamic safety velocity based on distance
            if pos_norm > 1000
                v_safe = 3.5;  % Allow higher velocity when far
            else
                v_safe = max(0.1, min(2.0, pos_norm / 1000));  % Less aggressive
            end
        
            % Check if velocity exceeds the dynamically computed safe velocity
            if speed > v_safe
                safe = false;  % Override only when absolutely necessary
                decel_mag = min(u_max_scalar, (speed - v_safe) * 2.0);
                safety_control(1:2) = -vel(1:2) / max(norm(vel(1:2)), 1e-6) * decel_mag;
                return;  % Exit function immediately if unsafe
            end

    % Progressive velocity profile based on distance
    if pos_norm > 2000
        v_safe = 2.5 * v_max;  
    elseif pos_norm > 1000
        v_safe = 1.5 * v_max;  
    elseif pos_norm > 500
        v_safe = 1.0 * v_max;  
    elseif pos_norm > 100
        v_safe = 0.5 * v_max;  
    elseif pos_norm > 10
        v_safe = 0.2 * v_max; 
    else
        v_safe = 0.05 * v_max; 
    end

    % Check approach angle for closing velocity
    pos_unit = pos / max(pos_norm, 1e-10);
    vel_proj = dot(vel, pos_unit);  % Projected velocity toward target

    % Allow positive velocity when far from the target
    if pos_norm > 500
        angle_safe = true;  
    else
        angle_safe = (vel_proj < 0) || (pos_norm < 5 && abs(vel_proj) < 0.01);
    end

    % Combined safety check
    magnitude_safe = speed <= v_safe;
    safe = magnitude_safe && angle_safe;

    % Calculate safety control if needed
    if ~safe
        % Direction for deceleration 
        if ~magnitude_safe
            decel_dir = -vel(1:2) / max(norm(vel(1:2)), 1e-10);
            decel_mag = min(u_max_scalar, (speed - v_safe) * 8.0);  
        else  % Not angle_safe
            decel_dir = pos_unit(1:2);
            decel_mag = min(u_max_scalar, abs(vel_proj) * 1.0);  
        end

        safety_control = decel_dir * decel_mag;
    end
end

% Initialize energy consumption tracking
energy = zeros(1, N);

% Initialize thrust magnitude and thruster status
thrust_magnitude = zeros(1, N-1);
thruster_status = zeros(1, N-1);

% Improved threshold for thrust activation
thrust_threshold = 0.01;  % Thrust threshold

% PWM counter initialization
pwm_counter = 1;

% Pontryagin's Minimum Principle optimization function
function u_pmp = pmp_optimize(x_state, A, B, Q, R, S, Np, Nc, u_min, u_max)
    % State and control dimensions
    nx = size(A,1);
    nu = size(B,2);
    
    % Terminal cost for final state
    lambda_T = S * x_state + 0.1 * [zeros(3,1); x_state(4:6)];
    
    % Initialize arrays
    lambda = zeros(nx, Np+1);
    lambda(:, Np+1) = lambda_T;
    
    % Backward integration of co-state equation
    for i = Np:-1:1
        lambda(:, i) = A' * lambda(:, i+1) + Q * x_state;
    end
    
    % Forward computation of optimal control
    x_traj = zeros(nx, Np+1);
    u_seq = zeros(nu, Np);
    x_traj(:, 1) = x_state;
    
    % Compute optimal control sequence
    for i = 1:Np
        % Minimize Hamiltonian: u* = -R^(-1) * B' * lambda
        u_i = -inv(R) * B' * lambda(:, i+1);
        
        % Apply control constraints
        u_i = min(max(u_i, u_min), u_max);
        
        % Update state using optimal control
        dist_effect = zeros(nx, 1);     
        x_traj(:, i+1) = A * x_traj(:, i) + B * u_i + dist_effect;
        u_seq(:, i) = u_i;
    end
    
    % Return first control action from sequence
    u_pmp = u_seq(:, 1);
end

% Lyapunov-based safety filter function
function [u_safe, V] = lyapunov_safety_filter(x_state, u_desired, A, B, P_lyap, u_min, u_max)
    % Compute current Lyapunov function value
    V = x_state' * P_lyap * x_state;
    
    % Compute derivative of V under desired control
    V_dot_desired = 2 * x_state' * P_lyap * (A*x_state + B*u_desired);
    
    % Check if desired control decreases Lyapunov function
    if V_dot_desired < -1e-6  % Small negative threshold for numerical stability
        % Desired control is safe
        u_safe = u_desired;
    else
        % QP formulation to find safe control
        H = eye(length(u_desired));
        f = -u_desired;
        
        % Constraint: V_dot < -epsilon
        A_con = 2 * x_state' * P_lyap * B;
        b_con = -2 * x_state' * P_lyap * A * x_state - 0.1;
        
        % Solve QP
        options = optimoptions('quadprog', 'Display', 'off');
        try
            u_safe = quadprog(H, f, A_con, b_con, [], [], u_min, u_max, u_desired, options);
        catch
            % Fallback if QP fails
            u_safe = u_min; % Conservative default
        end
    end
end

% MPC optimization function using Pontryagin's Minimum Principle
function [u_opt, Np_out, Nc_out] = mpc_optimize_pmp(x_current, A, B, Q, R, S, Np, Nc, x_min, x_max, u_min, u_max, v_max)
   
    % Get dimensions
    nx = size(A, 1);
    nu = size(B, 2);
    
    % Initialize prediction matrices
    F = zeros(nx * Np, nx);
    Phi = zeros(nx * Np, nu * Nc);
    
    % Construct prediction matrix F (free response)
    F(1:nx, :) = A;
    for i = 2:Np
        F((i-1)*nx+1:i*nx, :) = A * F((i-2)*nx+1:(i-1)*nx, :);
    end
    
    % Construct prediction matrix Phi (forced response)
    for i = 1:Np
        for j = 1:min(i, Nc)
            row_idx = (i-1)*nx+1:i*nx;
            col_idx = (j-1)*nu+1:j*nu;
            if i-j >= 0
                Phi(row_idx, col_idx) = A^(i-j) * B;
            end
        end
    end
    
       
    % Construct weight matrices
    Q_bar = zeros(nx * Np, nx * Np);
    for i = 1:Np-1
        Q_bar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx) = Q;
    end
    Q_bar((Np-1)*nx+1:Np*nx, (Np-1)*nx+1:Np*nx) = S;
    
    R_bar = zeros(nu * Nc, nu * Nc);
    for i = 1:Nc
        R_bar((i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu) = R;
    end
    
    % Compute QP matrices
    H = 2 * (Phi' * Q_bar * Phi + R_bar);
    H = (H + H')/2; % Ensure symmetry
    f = 2 * Phi' * Q_bar * (F * x_current);
    
    % Velocity constraints
    A_vel = zeros(Np*3, Nc*nu);
    b_vel = zeros(Np*3, 1);
    
    for i = 1:Np
        for j = 1:min(i, Nc)
            if i-j >= 0
                block = A^(i-j-1) * B;
                A_vel((i-1)*3+1:(i-1)*3+3, (j-1)*nu+1:(j-1)*nu+nu) = block(4:6, :);
            end
        end
        
        % Current predicted velocity without new control
        x_pred = A^i * x_current;
        for k = 1:i-1
            x_pred = x_pred + A^(i-k-1) * B * zeros(nu, 1);
        end
        
        % Adjust constraints based on predicted velocity
        b_vel((i-1)*3+1:(i-1)*3+3) = v_max - x_pred(4:6);
    end
    
    % Concatenate constraints
    A_con = [A_vel; -A_vel];
    b_con = [b_vel; b_vel + 2*v_max];
    
    % Solve QP problem
    options = optimoptions('quadprog', 'Display', 'off');
    try
        U = quadprog(H, f, A_con, b_con, [], [], repmat(u_min, Nc, 1), repmat(u_max, Nc, 1), [], options);
    catch
        % Fallback to simpler control if optimization fails
        U = zeros(nu * Nc, 1);
        
        pos = x_current(1:3);
        pos_norm = norm(pos);
        
        if pos_norm > 0
            dir = -pos(1:2) / pos_norm;
            U(1:2) = dir * min(0.5 * u_max_scalar, 0.1 * pos_norm);
        end
    end
    
    % Ensure U has at least nu elements
    if length(U) < nu
        U = [U; zeros(nu - length(U), 1)];
    end
    
    % Extract first control action
    u_opt = U(1:nu);
    
    % Return updated horizons 
    Np_out = Np;
    Nc_out = Nc;
end

% Adapt MPC weights based on phase and distance
function [Q_out, R_out] = adapt_weights(Q_in, R_in, distance, phase)
    % Base scaling factors
    pos_scale = 1.0;
    vel_scale = 1.0;
    control_scale = 1.0;

    % Apply distance-based scaling
    dist_factor = max(1.0, 100 / max(distance, 1.0)); % Increased sensitivity

    % Phase-specific tuning
    switch phase
        case 1 % Far approach
            pos_scale = pos_scale * 0.5;
            vel_scale = vel_scale * 0.5;
            control_scale = control_scale * 2.0; % Reduce energy
        case 2 % Mid-range
            pos_scale = pos_scale * 1.0;
            vel_scale = vel_scale * 1.0;
            control_scale = control_scale * 1.0;
        case 3 % Close approach
            pos_scale = pos_scale * 1.0;
            vel_scale = vel_scale * 2.0; % Increase velocity control
            control_scale = control_scale * 0.5;
        case 4 % Final docking
            pos_scale = pos_scale * 5.0;
            vel_scale = vel_scale * 5.0;  % More emphasis on velocity
            control_scale = control_scale * 0.1; % Min energy
    end

    % Apply distance factor
    pos_scale = pos_scale * dist_factor;
    vel_scale = vel_scale * dist_factor;

    % Apply scaling to weight matrices
    Q_out = Q_in;
    Q_out(1:3, 1:3) = Q_out(1:3, 1:3) * pos_scale;
    Q_out(4:6, 4:6) = Q_out(4:6, 4:6) * vel_scale;
    R_out = R_in * control_scale;
end

function u_terminal = terminal_guidance(x, A, B, u_max)
    pos = x(1:3);
    vel = x(4:6);
    pos_norm = norm(pos);
    vel_norm = norm(vel);
    
    % Zero-Effort-Miss / Zero-Effort-Velocity guidance
    % Estimate time-to-go based on current state
    t_go = max(30, min(100, pos_norm / max(0.05, vel_norm)));
    
    % Calculate zero-effort miss and velocity
    ZEM = pos + vel * t_go;
    ZEV = vel;
    
    % Guidance gains (time-varying)
    k_p = 6 / (t_go^2);
    k_v = 2 / t_go;
    
    % Calculate 3D acceleration command
    a_cmd = -k_p * ZEM - k_v * ZEV;
    
    % Project to control space (2D)
    u_terminal = a_cmd(1:2);
    
    % Apply constraints
    u_terminal = min(max(u_terminal, -u_max), u_max);
end

% Main Simulation Loop with Optimized Control Strategies
for k = 1:N-1
    % Get current state
    x_current = x(:,k);
    pos_norm = norm(x_current(1:3));
    vel_norm = norm(x_current(4:6));

    % Determine current mission phase based on distance
    if pos_norm > 2000
        phase = 1; % Far approach phase
        phase_name = 'Far Approach';
        control_mode = 'Far Approach (PMP-MPC Blend)';
    elseif pos_norm > 500  % Mid-range phase only when distance > 500 m
        phase = 2; % Mid-range phase
        phase_name = 'Mid-Range Approach';
        control_mode = 'Mid-Range (PMP-MPC Blend)';  % Update control mode
    elseif pos_norm > 20  % Close approach phase
        phase = 3; % Close approach phase
        phase_name = 'Close Approach';
        control_mode = 'Close Approach (MPC-Lyapunov Blend)';
    else
        phase = 4; % Final docking phase
        phase_name = 'Final Docking';
        control_mode = 'Final Docking (Terminal Guidance)';
    end

   % Square-root UKF implementation
    try
        sqrt_P = chol(P_ukf)'; % Cholesky decomposition 
    catch
        % Handle non-positive definite covariance (numerical issues)
        P_ukf = (P_ukf + P_ukf')/2; % Ensure symmetry
        P_ukf = P_ukf + 1e-6 * eye(size(P_ukf)); % Add small regularization
        sqrt_P = chol(P_ukf)';
    end

    % Generate sigma points
    X_sigma = zeros(L, 2*L+1);
    X_sigma(:, 1) = x_ukf; % Mean state
    for i = 1:L
        X_sigma(:, i+1) = x_ukf + gamma * sqrt_P(:, i); % Positive sigma points
        X_sigma(:, i+1+L) = x_ukf - gamma * sqrt_P(:, i); % Negative sigma points
    end

    % Propagate sigma points through the system dynamics
    for i = 1:2*L+1
        X_sigma(:, i) = A * X_sigma(:, i) + B * u(:,max(1,k-1)) + Ts;
    end

    % Calculate predicted mean with improved numerical stability
    x_pred = zeros(L, 1);
    weights_m = zeros(2*L+1, 1);
    weights_m(1) = lambda / (L + lambda);
    weights_m(2:end) = 1 / (2 * (L + lambda));

    for i = 1:2*L+1
        x_pred = x_pred + weights_m(i) * X_sigma(:, i);
    end

    % Calculate predicted covariance with improved stability
    P_pred = zeros(L, L);
    weights_c = weights_m;
    weights_c(1) = weights_c(1) + (1 - alpha^2 + beta); % Include beta for better handling of non-linearities

    for i = 1:2*L+1
        diff = X_sigma(:, i) - x_pred;
        P_pred = P_pred + weights_c(i) * (diff * diff');
    end
    P_pred = (P_pred + P_pred')/2; % Ensure symmetry
    P_pred = P_pred + Q; % Add process noise

    % UKF Update Step with phase-specific measurement noise
    switch phase
        case 1 % Far approach
            R_phase = R * 2.0; % Higher measurement uncertainty when far
        case 2 % Mid-range
            R_phase = R * 1.5;
        case 3 % Close approach
            R_phase = R * 1.0;
        case 4 % Final docking
            R_phase = R * 0.5; % Lower uncertainty for precision docking
    end

    % Simulate measurements with phase-specific noise
    z = x_current + mvnrnd(zeros(L, 1), R_phase)';

    % Generate sigma points for measurement using square-root form
    try
        S_pred = chol(P_pred, 'lower'); % Lower triangular Cholesky factor
    catch
        % Handle numerical issues
        P_pred = (P_pred + P_pred')/2; % Ensure symmetry
        P_pred = P_pred + 1e-6 * eye(size(P_pred)); % Add small regularization
        S_pred = chol(P_pred, 'lower');
    end

    Z_sigma = zeros(L, 2*L+1);
    Z_sigma(:, 1) = x_pred; % Mean state
    for i = 1:L
        Z_sigma(:, i+1) = x_pred + gamma * S_pred(:, i); % Positive sigma points
        Z_sigma(:, i+1+L) = x_pred - gamma * S_pred(:, i); % Negative sigma points
    end

    % Calculate predicted measurement
    z_pred = zeros(L, 1);
    for i = 1:2*L+1
        z_pred = z_pred + weights_m(i) * Z_sigma(:, i);
    end

    % Calculate measurement covariance with improved stability
    S = zeros(L, L);
    for i = 1:2*L+1
        diff = Z_sigma(:, i) - z_pred;
        S = S + weights_c(i) * (diff * diff');
    end
    S = (S + S')/2; % Ensure symmetry
    S = S + R_phase; % Add measurement noise

    % Calculate cross-covariance with improved stability
    C = zeros(L, L);
    for i = 1:2*L+1
        diff_x = X_sigma(:, i) - x_pred;
        diff_z = Z_sigma(:, i) - z_pred;
        C = C + weights_c(i) * (diff_x * diff_z');
    end

    % Kalman gain using more stable computation
    try
        K = C / S;
    catch
        % Fallback to pseudo-inverse for numerical stability
        K = C * pinv(S);
    end

    % Update state estimate with measurements
    x_ukf = x_pred + K * (z - z_pred);

    % Joseph form of covariance update for improved stability
    P_ukf = (eye(L) - K*eye(L)) * P_pred * (eye(L) - K*eye(L))' + K * R_phase * K';
    P_ukf = (P_ukf + P_ukf')/2; % Ensure symmetry

  % Simulate measurements with phase-specific noise
    z = x_current + mvnrnd(zeros(6, 1), R_phase)';

    % UKF update step
    [x_ukf, P_ukf] = ukf_update(x_ukf, P_ukf, u(:, k), z, Q, R, alpha, beta, kappa, Ts, A, B);

    % Store state values
    x(:, k+1) = x_ukf;
    position_history(:, k+1) = x_ukf(1:3);
    velocity_history(:, k+1) = x_ukf(4:6);

    % Check if docking is complete with tighter tolerances for ideal performance
    if pos_norm < 0.01 && vel_norm < 0.001
        disp(['Docking successfully completed at time: ', num2str(t(k+1)), ' seconds!']);
        disp(['Final position error: ', num2str(pos_norm), ' m']);
        disp(['Final velocity error: ', num2str(vel_norm), ' m/s']);
        u(:,k) = [0; 0];
        u_raw(:,k) = [0; 0];
        break;
    end

    % Coast phase logic
    if pos_norm < coast_distance_threshold && vel_norm < coast_velocity_threshold
        u(:, k) = [0; 0];
        u_raw(:, k) = [0; 0];
        control_mode = 'Coast Phase';
        coast_phase_active = true;
    else

      % Compute control strategies based on phase
      [Q_phase, R_phase, u_constraints, horizon_Np, horizon_Nc] = phase_specific_control(phase, pos_norm, vel_norm, Q_base, R_base, Np, Nc, u_max);
      [u_mpc, ~, ~] = mpc_optimize_pmp(x_current, A, B, Q_phase, R_phase, S, horizon_Np, horizon_Nc, x_min, x_max, u_min, u_constraints, v_max);

      % Check safety before applying final control
        % [safe, safety_control] = check_safety(x_current, v_max, u_max_scalar);
        % 
        % if ~safe
        %    u_raw(:,k) = 0.7 * u_mpc + 0.3 * safety_control;          % Override control with safe deceleration
        %     control_mode = 'Safety Override';
        % end

        % Activate terminal guidance during final docking phase
        if phase == 4
            u_terminal = terminal_guidance(x_current, A, B, u_max);
            u(:, k) = u_terminal;  % Use terminal guidance control inputs
        end

      % Compute PMP control for energy optimization
      u_pmp = pmp_optimize(x_ukf, A, B, Q_phase, R_phase, S_lyapunov, Np, Nc, u_min, u_constraints);

      % Compute Lyapunov-based safety control
      [u_lyap, V] = lyapunov_safety_filter(x_ukf, u_mpc, A, B, P, u_min, u_max);

      % Store Lyapunov value
      lyapunov_values(k) = V;

        % Phase-specific control blending
        switch phase
            case 1 % Far approach phase
                u_opt = 0.1 * u_pmp + 0.9 * u_mpc; % Prioritize MPC
                control_mode = 'Far Approach (MPC-Dominant)';
            case 2 % Mid-range phase
                u_opt = 0.2 * u_pmp + 0.8 * u_mpc; % More precise
                control_mode = 'Mid-Range (MPC-Controlled)';
            case 3 % Close approach phase
                u_opt = 0.1 * u_pmp + 0.9 * u_mpc; % High precision
                control_mode = 'Close Approach (MPC-Lyapunov Blend)';
            case 4 % Final docking phase
                u_opt = 0.7 * u_pmp + 0.3 * u_mpc; % Strict control
                control_mode = 'Final Docking (PMP Braking Mode)';
        end
      end

    
    % Apply Chebyshev filter for control signal smoothing
    if k > 5
        % Get control history for filtering
        u_history = u_raw(:, max(1, k-5):k-1);
        
        % Chebyshev filter parameters based on phase
        switch phase
            case 1 % Far approach
                filter_order = 2;
                ripple = 1.0;
            case 2 % Mid-range
                filter_order = 3;
                ripple = 0.8;
            case 3 % Close approach
                filter_order = 4;
                ripple = 0.5;
            case 4 % Final docking
                filter_order = 5;
                ripple = 0.3;
        end
        
        % Apply Chebyshev filter for control signal smoothing
        if k > 5
            % Get control history for filtering
            u_history = u_raw(:, max(1, k-5):k-1);

            % Chebyshev filter parameters based on phase
            switch phase
                case 1 % Far approach
                    filter_order = 2;
                    ripple = 1.0;
                case 2 % Mid-range
                    filter_order = 3;
                    ripple = 0.8;
                case 3 % Close approach
                    filter_order = 4;
                    ripple = 0.5;
                case 4 % Final docking
                    filter_order = 5;
                    ripple = 0.3;
            end

            % Apply Chebyshev filter
            [b, a] = cheby1(filter_order, ripple, 0.1, 'low');
            u_filtered = zeros(size(u_opt));
            for i = 1:length(u_opt)
                u_seq = [reshape(u_history(i,:), [], 1); u_opt(i)];
                u_filt = filter(b, a, u_seq);
                u_filtered(i) = u_filt(end);
            end

            % Blend filtered and raw control based on phase
            switch phase
                case 1
                    alpha_blend = 0.7;
                case 2
                    alpha_blend = 0.5;
                case 3
                    alpha_blend = 0.3;
                case 4
                    alpha_blend = 0.1;
            end

            u_raw(:,k) = alpha_blend * u_filtered + (1 - alpha_blend) * u_opt;
        else
            % Smooth control signal using Chebyshev filter
            u_raw(:,k) = apply_chebyshev_filter(u_opt, b_cheby, a_cheby);
        end

        % Store control history
        control_history(:, k) = u_raw(:, k);

        % Safety check with phase-specific velocity limits
        [safe, safety_control] = check_safety(x_current, v_max, u_max_scalar);
        safety_status(k) = safe;
        if ~safe
            u_raw(:,k) = safety_control;
            control_mode = 'Safety Override';
        end
    end

    % Apply dead-band and PWM for thruster control
    u_modified = enhanced_thruster_control(u_raw(:, k), pos_norm, vel_norm, pwm_counter, phase, u_max_scalar);
    u(:,k) = u_modified;

    % Update PWM counter
    pwm_counter = pwm_counter + 1;
    if pwm_counter > pwm_period
        pwm_counter = 1;
    end

    % Calculate thrust magnitude and energy
    thrust_mag = norm(u(:,k)) * m_chaser;
    thrust_magnitude(k) = thrust_mag;
    thruster_status(k) = (thrust_mag > thrust_threshold);

    % Update energy consumption with improved efficiency calculation
    if k > 1
        if thruster_status(k)
            efficiency_factor = 0.8 + 0.2 * (1 - thrust_mag / (u_max_scalar * m_chaser));
            energy(k+1) = energy(k) + (thrust_mag * norm(x_current(4:6)) * Ts) / efficiency_factor;
        else
            energy(k+1) = energy(k);
        end
    end

    % Record thruster firing points for analysis
    if thruster_status(k)
        thruster_firing_points = [thruster_firing_points, x_current(1:2)];
    end

    % Store control mode
    control_mode_history{k} = control_mode;

    % Display progress every 100 steps
    if mod(k, 100) == 0
        disp([ ...
            'Time: ', num2str(t(k)), ' s, Phase: ', phase_name, ...
            ', Distance: ', num2str(pos_norm), ' m, Velocity: ', num2str(vel_norm), ' m/s, Mode: ', control_mode]);
    end
end

function sigma_points_pred = predict_sigma_points(sigma_points, u, Ts)
    % System dimensions
    n = size(sigma_points, 1); % n = 6 (state dimension)
    m = size(u, 1); % m = 2 (control dimension)

    % State transition matrix (A) for orbital dynamics
    A = [1 0 0 Ts 0 0;
         0 1 0 0 Ts 0;
         0 0 1 0 0 Ts;
         0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1];

    % Input matrix (B) for control inputs
    B = [0 0;
         0 0;
         0 0;
         1 0;
         0 1;
         0 0];

    % Propagate sigma points through the system dynamics
    sigma_points_pred = A * sigma_points + B * u;
end

function z_pred_points = predict_measurements(sigma_points_pred)
    H = eye(6); % Identity matrix 
    % Predict measurements
    z_pred_points = H * sigma_points_pred;
end

function [x_est, P_est] = ukf_update(x_prev, P_prev, u, z, Q, R, alpha, beta, kappa, Ts, A, B)
    % State dimension
    n = length(x_prev);
    
    % Calculate UKF parameters
    lambda = alpha^2 * (n + kappa) - n;
    gamma = sqrt(n + lambda);
    
    % Weights calculation
    Wm = zeros(2*n+1, 1);
    Wc = zeros(2*n+1, 1);
    Wm(1) = lambda / (n + lambda);
    Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);
    for i = 2:2*n+1
        Wm(i) = 1 / (2*(n + lambda));
        Wc(i) = 1 / (2*(n + lambda));
    end
    
    % Ensure P_prev is positive definite
    P_prev = (P_prev + P_prev') / 2;  % Ensure symmetry
    P_prev = P_prev + 1e-6 * eye(n);  % Add small regularization
    
    % Generate sigma points
    sigma_points = zeros(n, 2*n+1);
    sigma_points(:,1) = x_prev;
    
    % Calculate square root of P using Cholesky decomposition
    sqrt_P = chol((n + lambda) * P_prev, 'lower');
    
    for i = 1:n
        sigma_points(:,i+1) = x_prev + sqrt_P(:,i);
        sigma_points(:,i+1+n) = x_prev - sqrt_P(:,i);
    end
    
    % Prediction step
    sigma_points_pred = zeros(n, 2*n+1);
    for i = 1:2*n+1
        % Propagate each sigma point through the dynamics model
        sigma_points_pred(:,i) = A * sigma_points(:,i) + B * u;
    end
    
    % Calculate predicted mean
    x_pred = zeros(n, 1);
    for i = 1:2*n+1
        x_pred = x_pred + Wm(i) * sigma_points_pred(:,i);
    end
    
    % Calculate predicted covariance
    P_pred = Q;  % Start with process noise
    for i = 1:2*n+1
        diff = sigma_points_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i) * (diff * diff');
    end
    
    % Update step with measurements
    H = eye(n);  % Replace with actual measurement matrix if different
    z_pred = zeros(length(z), 2*n+1);
    for i = 1:2*n+1
        z_pred(:,i) = H * sigma_points_pred(:,i);  % Apply measurement model
    end
    
    % Predicted measurement
    z_mean = zeros(length(z), 1);
    for i = 1:2*n+1
        z_mean = z_mean + Wm(i) * z_pred(:,i);
    end
    
    % Innovation covariance
    S = R;  % Start with measurement noise
    for i = 1:2*n+1
        diff = z_pred(:,i) - z_mean;
        S = S + Wc(i) * (diff * diff');
    end
    
    % Cross correlation matrix
    Pxz = zeros(n, length(z));
    for i = 1:2*n+1
        diff_x = sigma_points_pred(:,i) - x_pred;
        diff_z = z_pred(:,i) - z_mean;
        Pxz = Pxz + Wc(i) * (diff_x * diff_z');
    end
    
    % Kalman gain (using pseudoinverse for stability)
    K = Pxz * pinv(S);
    
    % State and covariance update (Joseph form for stability)
    I = eye(n);
    P_est = (I - K * H) * P_pred * (I - K * H)' + K * R * K';
    x_est = x_pred + K * (z - z_mean);
end

% ***** Performance Metrics *****
% Calculate Total Maneuver Time
total_maneuver_time = t(k+1);

% Calculate Final Position and Velocity Error
final_position_error = norm(x(1:3, k+1));
final_velocity_error = norm(x(4:6, k+1));

% Calculate Total Energy Consumed
total_energy_consumed = energy(k+1);

% Calculate Maximum Thrust Magnitude
max_thrust_magnitude = max(thrust_magnitude);

% Calculate Total Thruster On Time
total_thruster_on_time = sum(thruster_status) * Ts;
thruster_on_percentage = (total_thruster_on_time / total_maneuver_time) * 100;

% Display Performance Metrics
disp(' ');
disp('***** Performance Metrics *****');
disp(['Total Maneuver Time: ', num2str(total_maneuver_time), ' seconds']);
disp(['Final Position Error: ', num2str(final_position_error), ' m']);
disp(['Final Velocity Error: ', num2str(final_velocity_error), ' m/s']);
disp(['Total Energy Consumed: ', num2str(total_energy_consumed), ' J']);
disp(['Maximum Thrust Magnitude: ', num2str(max_thrust_magnitude), ' N']);
disp(['Total Thruster On Time: ', num2str(total_thruster_on_time), ' seconds (', num2str(thruster_on_percentage), '%)']);

% ***** Plotting *****
% Figure 1: R-bar Mode Docking Simulation
figure('Name', 'R-bar Mode Docking Simulation');
subplot(2,2,1);
plot(x(1,1:k), x(2,1:k), 'b-', 'LineWidth', 2);
hold on;
plot(0, 0, 'r*', 'MarkerSize', 10);
plot(x(1,1), x(2,1), 'go', 'MarkerSize', 10);
grid on;
xlabel('V-bar (m)');
ylabel('R-bar (m)');
title('Approach Trajectory');
legend('Trajectory', 'Target', 'Start', 'Location', 'best');

subplot(2,2,2);
plot(t(1:k+1), sqrt(sum(x(1:3,1:k+1).^2)), 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Distance (m)');
title('Distance to Target');

subplot(2,2,3);
plot(t(1:k+1), x(4,1:k+1), 'r-', 'LineWidth', 2, 'DisplayName', 'V_x');
hold on;
plot(t(1:k+1), x(5,1:k+1), 'b-', 'LineWidth', 2, 'DisplayName', 'V_y');
plot(t(1:k+1), sqrt(x(4,1:k+1).^2 + x(5,1:k+1).^2), 'k--', 'LineWidth', 1, 'DisplayName', 'V_{mag}');
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Velocity Components');
legend('Location', 'best');

subplot(2,2,4);
stairs(t(1:k), u(1,1:k), 'r-', 'LineWidth', 2, 'DisplayName', 'u_x');
hold on;
stairs(t(1:k), u(2,1:k), 'b-', 'LineWidth', 2, 'DisplayName', 'u_y');
stairs(t(1:k), sqrt(u(1,1:k).^2 + u(2,1:k).^2), 'k--', 'LineWidth', 1, 'DisplayName', 'u_{mag}');
grid on;
xlabel('Time (s)');
ylabel('Control Input (m/s^2)');
title('Control Inputs');
legend('Location', 'best');

% Figure 2: Thrust Comparison
figure('Name', 'Thrust Comparison');
subplot(2,1,1);
stairs(t(1:k), u_raw(1,1:k), 'r--', 'LineWidth', 1, 'DisplayName', 'Raw u_x');
hold on;
stairs(t(1:k), u_raw(2,1:k), 'b--', 'LineWidth', 1, 'DisplayName', 'Raw u_y');
stairs(t(1:k), u(1,1:k), 'r-', 'LineWidth', 2, 'DisplayName', 'PWM u_x');
stairs(t(1:k), u(2,1:k), 'b-', 'LineWidth', 2, 'DisplayName', 'PWM u_y');
grid on;
xlabel('Time (s)');
ylabel('Control Input (m/s^2)');
title('Raw vs PWM Control Inputs');
legend('Location', 'best');

subplot(2,1,2);
plot(t(1:k), thrust_magnitude(1:k), 'g-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Thrust Magnitude (N)');
title('Thrust Magnitude');

% Figure 3: Thruster Activity
figure('Name', 'Thruster Activity');
subplot(2,1,1);
stem(t(1:k), thruster_status(1:k), 'k-', 'LineWidth', 1, 'Marker', 'none');
grid on;
xlabel('Time (s)');
ylabel('Thruster Status');
title('Thruster On/Off Status');
ylim([-0.1 1.1]);

subplot(2,1,2);
cumulative_on_time = cumsum(thruster_status(1:k)) * Ts;
plot(t(1:k), cumulative_on_time, 'm-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Cumulative On Time (s)');
title('Cumulative Thruster On Time');

% Figure 4: Energy & Approach Analysis
figure('Name', 'Energy & Approach Analysis');
subplot(2,1,1);
plot(t(1:k+1), energy(1:k+1), 'm-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Energy Consumption (J)');
title('Energy Consumption Over Time');

subplot(2,1,2);
plot(sqrt(sum(x(1:3,1:k+1).^2)), sqrt(sum(x(4:6,1:k+1).^2)), 'c-', 'LineWidth', 2);
grid on;
xlabel('Distance to Target (m)');
ylabel('Approach Speed (m/s)');
title('Approach Speed vs Distance');

% Figure 5: Trajectory and Thruster Firing Points
figure('Name', 'Trajectory and Thruster Firing Points');
plot(x(1,1:k+1), x(2,1:k+1), 'b-', 'LineWidth', 2);  % Chaser trajectory
hold on;
plot(0, 0, 'r*', 'MarkerSize', 10);  % Target position
plot(x(1,1), x(2,1), 'go', 'MarkerSize', 10);  % Chaser starting position

% Plot thruster firing points
if ~isempty(thruster_firing_points)
    plot(thruster_firing_points(1,:), thruster_firing_points(2,:), 'rx', 'MarkerSize', 3);
end

grid on;
xlabel('V-bar (m)');
ylabel('R-bar (m)');
title('Trajectory and Thruster Firing Points');
legend('Chaser Trajectory', 'Target', 'Chaser Start', 'Thruster Firing', 'Location', 'best');