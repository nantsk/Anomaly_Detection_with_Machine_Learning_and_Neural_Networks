% R-bar Mode with Model Predictive Control (MPC) - Optimized with Real-Time Adaptation
clear;
close all;
clc;

% Orbital parameters
alt = 450e3;            % Altitude of orbit (m)
Re = 6371e3;            % Earth radius (m)
mu = 3.986004418e14;    % Earth gravitational parameter (m^3/s^2)
r_orbit = Re + alt;     % Orbital radius (m)
n = sqrt(mu/r_orbit^3); % Mean motion (rad/s)

% Satellite parameters
m_target = 150;         % Target satellite mass (kg)
m_chaser = 350;         % Chaser satellite mass (kg)
thrust_x = 17;          % Total thrust (N) - 17 N per thruster
num_thrusters = 4;      % Number of thrusters

% MPC parameters
Ts = 0.5;               % Sampling time (s)
Np = 50;                % Increased prediction horizon
Nc = 10;                % Increased control horizon
Q_base = diag([1e6, 1e6, 1e6, 1e5, 1e5, 1e5]);  % Increased position weights
R_base = diag([0.1, 0.1]);  % Reduced control weights
S = 1e8 * eye(6);       % Increased terminal state weighting matrix

% State and input constraints
u_max = ((thrust_x * num_thrusters) / m_chaser) * 0.2;  % Reduced thrust limit (m/s^2)
v_max = 2.0;                 % Reduced maximum velocity (m/s)
x_min = [-inf; -inf; -inf; -v_max; -v_max; -v_max];  % State constraints
x_max = [inf; inf; inf; v_max; v_max; v_max];
u_min = [-u_max; -u_max];     % Control input constraints
u_max = [u_max; u_max];

% Initial conditions
x0 = [-2903.1; -2991.5; 0; 1; 1; 0];  % 2.9031 km behind, 2.9915 km below
t_final = 2900;               % Simulation time (s)
t = 0:Ts:t_final;
N = length(t);

% Initialize state and control vectors
x = zeros(6, N);
u = zeros(2, N-1);
x(:,1) = x0;

% Initialize thruster firing points
thruster_firing_points = [];

% Continuous-time state-space matrices
A_c = [0 0 0 1 0 0;
       0 0 0 0 1 0;
       0 0 0 0 0 1;
       3*n^2 0 0 0 2*n 0;
       0 0 0 -2*n 0 0;
       0 0 -n^2 0 0 0];
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

% Custom QP solver using quadprog for better performance
function u_opt = custom_qp_solver(H, f, lb, ub)
    options = optimoptions('quadprog', 'Display', 'off');
    u_opt = quadprog(H, f, [], [], [], [], lb, ub, [], options);
end

% MPC optimization function with real-time adaptation
function [u_opt, Np, Nc] = mpc_optimize(x_current, A, B, Q_base, R_base, S, Np, Nc, x_min, x_max, u_min, u_max, v_max)
    nx = size(A,1);
    nu = size(B,2);
    
    % Adaptive horizons based on distance
    distance = norm(x_current(1:3));
    if distance > 2000  % Far away
        Np = 50; Nc = 10;
    elseif distance > 500
        Np = 30; Nc = 5;
    else
        Np = 20; Nc = 3;
    end
    
    % Real-time adaptation of Q and R based on distance and velocity
    vel_norm = norm(x_current(4:6));
    Q = Q_base * (1 + 10 / (1 + distance));  % Increase position weights when far away
    R = R_base * (1 + vel_norm / v_max);     % Increase control weights at high velocity
    
    H = zeros(Nc*nu, Nc*nu);
    f = zeros(Nc*nu, 1);
    
    Phi = zeros(nx*Np, nx);
    Gamma = zeros(nx*Np, nu*Nc);
    
    temp = eye(nx);
    for i = 1:Np
        Phi((i-1)*nx+1:i*nx,:) = temp*A;
        temp = temp*A;
    end
    
    for i = 1:Np
        for j = 1:min(i,Nc)
            if i-j+1 > 0
                Gamma((i-1)*nx+1:i*nx,(j-1)*nu+1:j*nu) = A^(i-j)*B;
            end
        end
    end
    
    Q_bar = kron(eye(Np), Q);
    R_bar = kron(eye(Nc), R);
    
    Q_bar(end-nx+1:end,end-nx+1:end) = S;
    
    H = Gamma'*Q_bar*Gamma + R_bar;
    H = (H + H')/2; 
    
    f = Gamma'*Q_bar*Phi*x_current;
    
    % Add energy term to the cost function
    energy_weight = 0.2;  % Increased energy weight
    H = H + energy_weight * eye(size(H));
    
    lb = repmat(u_min, Nc, 1);
    ub = repmat(u_max, Nc, 1);
    
    u_sequence = custom_qp_solver(H, f, lb, ub);
    
    if ~isempty(u_sequence)
        u_opt = reshape(u_sequence(1:nu), [nu, 1]);
    else
        u_opt = zeros(nu, 1);
    end
end

% Enhanced safety check function
function safe = check_safety(x, v_max, d_safe)
    pos = x(1:3);
    vel = x(4:6);
    dist = norm(pos);
    speed = norm(vel);
    
    if dist > 2000  % Far away
        safe = speed <= v_max;
    elseif dist > 500  % Medium range
        safe = speed <= min(v_max, dist/50);  % Allow higher speed
    else  % Close range
        safe = (speed <= min(v_max, dist/10)) && (dist >= d_safe || speed <= 0.05);
    end
end

% Initialize energy consumption tracking
energy = zeros(1, N);

% Initialize thrust magnitude and thruster status
thrust_magnitude = zeros(1, N-1);
thruster_status = zeros(1, N-1);

% Threshold for thrust activation
thrust_threshold = 1e-6;  % Thrust below this value is considered zero

% Main simulation loop with real-time adaptation
for k = 1:N-1
    x_current = x(:,k);
    [u_opt, Np, Nc] = mpc_optimize(x_current, A, B, Q_base, R_base, S, Np, Nc, x_min, x_max, u_min, u_max, v_max);
    
    if ~check_safety(x_current, v_max, 25)
        pos = x_current(1:3);
        vel = x_current(4:6);
        u_opt = -0.5 * vel(1:2);  % Smooth deceleration
    end
    
    u(:,k) = u_opt;
    x(:,k+1) = A*x(:,k) + B*u(:,k);
    
    F = u(:,k) * m_chaser;  
    v = x(4:6,k);           
    energy(k+1) = energy(k) + norm(F) * norm(v) * Ts;  
    
    thrust_magnitude(k) = norm(F);
    
    % Update thruster status based on thrust magnitude
    thruster_status(k) = norm(F) > thrust_threshold;
    
    if thruster_status(k)
        thruster_firing_points = [thruster_firing_points, x(1:3,k+1)];
    end
    
    % Check if docking is complete (distance < 0.1 and velocity < v_min)
    if norm(x(1:3,k+1)) < 0.1 && norm(x(4:6,k+1)) < 0.05
        disp('Docking complete!');
        break;
    end
end

% Plotting and performance metrics

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
plot(t(1:k), sqrt(sum(x(1:3,1:k).^2)), 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Distance (m)');
title('Distance to Target');

subplot(2,2,3);
plot(t(1:k), x(4,1:k), 'r-', 'LineWidth', 2, 'DisplayName', 'V_x');
hold on;
plot(t(1:k), x(5,1:k), 'b-', 'LineWidth', 2, 'DisplayName', 'V_y');
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Velocity Components');
legend('Location', 'best');

subplot(2,2,4);
stairs(t(1:k-1), u(1,1:k-1), 'r-', 'LineWidth', 2, 'DisplayName', 'u_x');
hold on;
stairs(t(1:k-1), u(2,1:k-1), 'b-', 'LineWidth', 2, 'DisplayName', 'u_y');
grid on;
xlabel('Time (s)');
ylabel('Control Input (m/s^2)');
title('Control Inputs');
legend('Location', 'best');

% Figure 2: Torque Components
figure('Name', 'Torque Components');
plot(t(1:k-1), u(1,1:k-1) * m_chaser, 'r-', 'LineWidth', 2, 'DisplayName', 'Torque_x');
hold on;
plot(t(1:k-1), u(2,1:k-1) * m_chaser, 'b-', 'LineWidth', 2, 'DisplayName', 'Torque_y');
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torque Components');
legend('Location', 'best');

% Figure 3: Thruster Activity
figure('Name', 'Thruster Activity');
subplot(2,1,1);
plot(t(1:k-1), thrust_magnitude(1:k-1), 'g-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Thrust Magnitude (N)');
title('Thrust Magnitude');

subplot(2,1,2);
stem(t(1:k-1), thruster_status(1:k-1), 'k-', 'LineWidth', 2, 'Marker', 'none');
grid on;
xlabel('Time (s)');
ylabel('Thruster Status');
title('Thruster On/Off Status');
ylim([-0.1 1.1]);

% Figure 4: Energy & Time Analysis
figure('Name', 'Energy & Time Analysis');
subplot(2,1,1);
plot(t(1:k), energy(1:k), 'm-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Energy Consumption (J)');
title('Energy Consumption Over Time');

subplot(2,1,2);
plot(sqrt(sum(x(1:3,1:k).^2)), sqrt(sum(x(4:6,1:k).^2)), 'c-', 'LineWidth', 2);
grid on;
xlabel('Distance to Target (m)');
ylabel('Approach Speed (m/s)');
title('Approach Speed vs Distance');

% Print summary statistics
fprintf('\nPerformance Metrics:\n');
fprintf('Total Maneuver Time: %.2f seconds\n', t(k));
fprintf('Final Position Error: %.3f m\n', norm(x(1:3,k)));
fprintf('Final Velocity Error: %.3f m/s\n', norm(x(4:6,k)));
fprintf('Total Energy Consumed: %.3f J\n', energy(k));
fprintf('Maximum Thrust Magnitude: %.3f N\n', max(thrust_magnitude(1:k-1)));
fprintf('Total Thruster On Time: %.2f seconds\n', sum(thruster_status(1:k-1)) * Ts);

% Figure 5: Trajectory and Thruster Firing Points
figure('Name', 'Trajectory and Thruster Firing Points');
plot(x(1,1:k), x(2,1:k), 'b-', 'LineWidth', 2);  % Chaser trajectory
hold on;
plot(0, 0, 'r*', 'MarkerSize', 10);  % Target position
plot(x(1,1), x(2,1), 'go', 'MarkerSize', 10);  % Chaser starting position

% Plot thruster firing points
if ~isempty(thruster_firing_points)
    plot(thruster_firing_points(1,:), thruster_firing_points(2,:), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
end

grid on;
xlabel('V-bar (m)');
ylabel('R-bar (m)');
title('Trajectory and Thruster Firing Points');
legend('Chaser Trajectory', 'Target', 'Chaser Start', 'Thruster Firing', 'Location', 'best');