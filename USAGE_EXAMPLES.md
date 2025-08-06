# Berkeley SciComp Framework - Usage Examples

This document provides comprehensive examples of how to use the various modules in the Berkeley SciComp framework across Python, MATLAB, and Mathematica platforms.

## Table of Contents

1. [Signal Processing](#signal-processing)
2. [Stochastic Processes](#stochastic-processes)
3. [Optimization](#optimization)
4. [Quantum Physics](#quantum-physics)
5. [Machine Learning](#machine-learning)
6. [Cross-Platform Integration](#cross-platform-integration)

---

## Signal Processing

### Python Implementation

```python
from Python.Signal_Processing import SignalProcessor, SpectralAnalyzer
import numpy as np
import matplotlib.pyplot as plt

# Initialize processor
fs = 1000  # Sampling rate
processor = SignalProcessor(sampling_rate=fs)
analyzer = SpectralAnalyzer(sampling_rate=fs)

# Generate multi-component signal with noise
duration = 2.0
t, signal = processor.generate_signal('multi_sine', duration,
                                     frequency=[50, 120, 200],
                                     amplitude=[1.0, 0.5, 0.3],
                                     noise_level=0.2)

# Spectral analysis
freq, fft_mag = processor.compute_fft(signal, window='hann')
f_psd, Pxx = analyzer.compute_power_spectrum(signal, method='welch')

# Filter design and application
b, a = processor.design_filter('bandpass', [40, 150], order=4)
filtered = processor.apply_filter(signal, b, a)

# Feature extraction
features = processor.extract_features(filtered)
spectral_features = analyzer.compute_spectral_features(filtered)

print(f"Dominant frequency: {features['dominant_frequency']:.2f} Hz")
print(f"Spectral centroid: {spectral_features['spectral_centroid']:.2f} Hz")
```

### MATLAB Implementation

```matlab
% Initialize Signal Processor
processor = signal_processing.SignalProcessor(1000);

% Generate test signal
[t, signal] = processor.generateSignal('multi_sine', 2.0, ...
    'frequency', [50, 120, 200], ...
    'amplitude', [1.0, 0.5, 0.3], ...
    'noise_level', 0.2);

% Compute FFT and PSD
[freq, mag] = processor.computeFFT(signal, 'window', 'hann');
[f, Pxx] = processor.computePSD(signal, 'method', 'welch');

% Filter design and application
[b, a] = processor.designFilter('bandpass', [40, 150], 'order', 4);
filtered = processor.applyFilter(signal, b, a);

% Feature extraction
features = processor.extractFeatures(filtered);

fprintf('Dominant frequency: %.2f Hz\n', features.dominant_frequency);
fprintf('Spectral centroid: %.2f Hz\n', features.spectral_centroid);

% Visualization
processor.plotSignal(t, signal, 'title', 'Multi-component Signal');
processor.plotSpectrum(freq, mag, 'title', 'Frequency Spectrum');
```

---

## Stochastic Processes

### Brownian Motion and SDEs

```python
from Python.Stochastic.stochastic_processes import (
    BrownianMotion, OrnsteinUhlenbeck, StochasticDifferentialEquation
)
import matplotlib.pyplot as plt

# Geometric Brownian Motion (for stock prices)
gbm = BrownianMotion(drift=0.05, volatility=0.2, seed=42)
t, S = gbm.generate_geometric_path(T=252/252, n_steps=252, S0=100)

plt.figure(figsize=(12, 8))

# Plot 1: Stock price simulation
plt.subplot(2, 2, 1)
plt.plot(t * 252, S)
plt.title('Geometric Brownian Motion (Stock Price)')
plt.xlabel('Days')
plt.ylabel('Price ($)')

# Mean-reverting process (for interest rates)
ou = OrnsteinUhlenbeck(theta=2.0, mu=0.03, sigma=0.01, seed=42)
t_ou, r = ou.generate_path(T=5.0, n_steps=1000, x0=0.05)

plt.subplot(2, 2, 2)
plt.plot(t_ou, r * 100)  # Convert to percentage
plt.axhline(y=3, color='r', linestyle='--', label='Long-term mean')
plt.title('Mean-Reverting Interest Rate')
plt.xlabel('Years')
plt.ylabel('Rate (%)')
plt.legend()

# Custom SDE solution
sde = StochasticDifferentialEquation(seed=42)

# Stochastic volatility model: dV = κ(θ - V)dt + σ√V dW
def drift(v, t): return 2.0 * (0.04 - v)
def diffusion(v, t): return 0.3 * np.sqrt(max(v, 0))

t_vol, V = sde.euler_maruyama(drift, diffusion, x0=0.04, T=2.0, n_steps=500)

plt.subplot(2, 2, 3)
plt.plot(t_vol, V * 100)
plt.title('Stochastic Volatility (CIR Model)')
plt.xlabel('Years')
plt.ylabel('Volatility (%)')

# 2D Random Walk
from Python.Stochastic.stochastic_processes import RandomWalk
rw = RandomWalk(seed=42)
x, y = rw.simple_walk_2d(n_steps=1000)

plt.subplot(2, 2, 4)
plt.plot(x, y, alpha=0.7, linewidth=0.8)
plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
plt.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
plt.title('2D Random Walk')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()

plt.tight_layout()
plt.show()
```

### Monte Carlo Option Pricing

```python
import numpy as np
from scipy.stats import norm

def monte_carlo_option_pricing():
    """European call option pricing using geometric Brownian motion."""
    
    # Option parameters
    S0 = 100.0    # Initial stock price
    K = 105.0     # Strike price
    T = 1.0       # Time to maturity
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    n_paths = 10000
    
    # Monte Carlo simulation
    payoffs = []
    for i in range(n_paths):
        gbm = BrownianMotion(drift=r, volatility=sigma, seed=i)
        _, S = gbm.generate_geometric_path(T, 252, S0)
        payoff = max(S[-1] - K, 0)
        payoffs.append(payoff)
    
    # Option price
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    mc_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    # Black-Scholes analytical solution
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    print(f"Monte Carlo Price: ${mc_price:.4f} ± ${1.96*mc_std:.4f}")
    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Difference: ${abs(mc_price - bs_price):.4f}")

monte_carlo_option_pricing()
```

---

## Optimization

### Multi-Objective Optimization

```python
from Python.Optimization.multi_objective import ParetoOptimizer
from Python.Optimization.unconstrained import BFGS
import numpy as np

# Define multi-objective problem (minimize both objectives)
def objective1(x):
    return x[0]**2 + x[1]**2

def objective2(x):
    return (x[0] - 1)**2 + (x[1] - 1)**2

def grad1(x):
    return np.array([2*x[0], 2*x[1]])

def grad2(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 1)])

# Pareto optimization
pareto_opt = ParetoOptimizer(n_objectives=2)
pareto_front = pareto_opt.weighted_sum_method(
    objectives=[objective1, objective2],
    gradients=[grad1, grad2],
    bounds=[(-2, 2), (-2, 2)],
    n_points=50
)

print(f"Found {len(pareto_front)} Pareto optimal solutions")

# Single objective optimization
optimizer = BFGS(max_iterations=1000, tolerance=1e-6)
result = optimizer.minimize(objective1, np.array([2.0, 2.0]), gradient=grad1)

print(f"Single objective optimum: {result.x}")
print(f"Function value: {result.fun:.6f}")
print(f"Converged: {result.success}")
```

### MATLAB Optimization

```matlab
% Initialize optimizers
gd = optimization.GradientDescent('MaxIterations', 1000, 'Tolerance', 1e-6);
bfgs = optimization.BFGS('MaxIterations', 500);
sa = optimization.SimulatedAnnealing('MaxIterations', 5000);

% Define Rosenbrock function
rosenbrock = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
rosenbrock_grad = @(x) [-400*x(1)*(x(2) - x(1)^2) - 2*(1 - x(1)); 
                        200*(x(2) - x(1)^2)];

% Initial point
x0 = [-1.2; 1.0];

% Solve with different methods
result_gd = gd.minimize(rosenbrock, x0, rosenbrock_grad);
result_bfgs = bfgs.minimize(rosenbrock, x0, rosenbrock_grad);

% Global optimization for Ackley function
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - ...
              exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;

bounds = {[-5, 5], [-5, 5]};
result_sa = sa.minimize(ackley, bounds);

fprintf('Gradient Descent: x = [%.4f, %.4f], f = %.8f\n', ...
        result_gd.x(1), result_gd.x(2), result_gd.fun);
fprintf('BFGS: x = [%.4f, %.4f], f = %.8f\n', ...
        result_bfgs.x(1), result_bfgs.x(2), result_bfgs.fun);
fprintf('Simulated Annealing: x = [%.4f, %.4f], f = %.8f\n', ...
        result_sa.x(1), result_sa.x(2), result_sa.fun);
```

---

## Quantum Physics

### Quantum Harmonic Oscillator

```python
from Python.quantum_physics.quantum_dynamics.harmonic_oscillator import QuantumHarmonic
import numpy as np
import matplotlib.pyplot as plt

# Initialize quantum harmonic oscillator
qho = QuantumHarmonic(omega=1.0, n_max=20, x_min=-5, x_max=5, n_points=1000)

# Compute eigenstates and eigenvalues
eigenvalues = qho.eigenvalues()
psi_0 = qho.eigenstate(n=0)  # Ground state
psi_1 = qho.eigenstate(n=1)  # First excited state

# Position representation
x = qho.position_grid()

# Probability densities
prob_0 = np.abs(psi_0)**2
prob_1 = np.abs(psi_1)**2

# Time evolution
t_points = np.linspace(0, 4*np.pi, 100)
coherent_evolution = qho.coherent_state_evolution(alpha=2.0, t_points=t_points)

plt.figure(figsize=(15, 5))

# Plot eigenstates
plt.subplot(1, 3, 1)
plt.plot(x, prob_0, 'b-', label='Ground state |0⟩', linewidth=2)
plt.plot(x, prob_1, 'r-', label='First excited |1⟩', linewidth=2)
plt.xlabel('Position (x)')
plt.ylabel('Probability density |ψ(x)|²')
plt.title('Harmonic Oscillator Eigenstates')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot potential and energy levels
plt.subplot(1, 3, 2)
V = 0.5 * qho.omega**2 * x**2
plt.plot(x, V, 'k-', label='Potential V(x)', linewidth=2)
for n in range(5):
    E_n = eigenvalues[n]
    plt.axhline(y=E_n, color='blue', linestyle='--', alpha=0.7)
    plt.text(3, E_n, f'E_{n} = {E_n:.1f}')
plt.xlabel('Position (x)')
plt.ylabel('Energy')
plt.title('Energy Levels')
plt.ylim(0, 5)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot coherent state time evolution
plt.subplot(1, 3, 3)
times = t_points[:50:5]  # Select every 5th time point
for i, t in enumerate(times):
    psi_t = coherent_evolution[i*5]
    prob_t = np.abs(psi_t)**2
    plt.plot(x, prob_t + i*0.1, alpha=0.7, linewidth=1)
plt.xlabel('Position (x)')
plt.ylabel('Probability density (offset)')
plt.title('Coherent State Evolution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Ground state energy: E_0 = {eigenvalues[0]:.4f}")
print(f"First excited energy: E_1 = {eigenvalues[1]:.4f}")
print(f"Energy spacing: ΔE = {eigenvalues[1] - eigenvalues[0]:.4f}")
```

---

## Machine Learning

### Physics-Informed Neural Networks (PINNs)

```python
from Python.ml_physics.pinns.heat_equation_pinn import HeatEquationPINN
import numpy as np
import tensorflow as tf

# Define heat equation PINN
pinn = HeatEquationPINN(
    layers=[2, 50, 50, 50, 1],  # Network architecture
    alpha=1.0,  # Thermal diffusivity
    domain={'x': [0, 1], 't': [0, 1]}
)

# Initial condition: u(x, 0) = sin(πx)
def initial_condition(x):
    return np.sin(np.pi * x)

# Boundary conditions: u(0, t) = u(1, t) = 0
def boundary_condition_left(t):
    return 0.0

def boundary_condition_right(t):
    return 0.0

# Training data
n_train = 1000
x_train = np.random.uniform(0, 1, (n_train, 1))
t_train = np.random.uniform(0, 1, (n_train, 1))

# Initial condition data
n_ic = 100
x_ic = np.linspace(0, 1, n_ic).reshape(-1, 1)
t_ic = np.zeros((n_ic, 1))
u_ic = initial_condition(x_ic)

# Boundary condition data
n_bc = 50
t_bc = np.linspace(0, 1, n_bc).reshape(-1, 1)
x_bc_left = np.zeros((n_bc, 1))
x_bc_right = np.ones((n_bc, 1))
u_bc_left = np.zeros((n_bc, 1))
u_bc_right = np.zeros((n_bc, 1))

# Train the PINN
history = pinn.train(
    x_interior=x_train, t_interior=t_train,
    x_ic=x_ic, t_ic=t_ic, u_ic=u_ic,
    x_bc_left=x_bc_left, t_bc=t_bc, u_bc_left=u_bc_left,
    x_bc_right=x_bc_right, t_bc=t_bc, u_bc_right=u_bc_right,
    epochs=5000
)

# Prediction
x_test = np.linspace(0, 1, 100)
t_test = np.linspace(0, 1, 100)
X_test, T_test = np.meshgrid(x_test, t_test)
X_flat = X_test.flatten().reshape(-1, 1)
T_flat = T_test.flatten().reshape(-1, 1)

U_pred = pinn.predict(X_flat, T_flat)
U_pred = U_pred.reshape(X_test.shape)

print(f"PINN training completed. Final loss: {history.history['loss'][-1]:.6f}")
```

---

## Cross-Platform Integration

### Calling MATLAB from Python

```python
# Example of using MATLAB engine from Python
import matlab.engine
import numpy as np

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Add MATLAB Signal Processing toolbox to path
eng.addpath('MATLAB/Signal_Processing', nargout=0)

# Generate signal in MATLAB
eng.workspace['fs'] = 1000.0
eng.eval("processor = signal_processing.SignalProcessor(fs);", nargout=0)
eng.eval("[t, signal] = processor.generateSignal('sine', 1.0, 'frequency', 50);", nargout=0)

# Get results back to Python
t_matlab = np.array(eng.workspace['t']).flatten()
signal_matlab = np.array(eng.workspace['signal']).flatten()

# Process in Python
from Python.Signal_Processing import SignalProcessor
processor_py = SignalProcessor(sampling_rate=1000)
freq, mag = processor_py.compute_fft(signal_matlab)

print(f"Signal generated in MATLAB, processed in Python")
print(f"Peak frequency: {freq[np.argmax(mag[1:]) + 1]:.2f} Hz")

# Stop MATLAB engine
eng.quit()
```

### Mathematica Integration Example

```mathematica
(* Load Berkeley SciComp packages *)
<< "Mathematica/Signal_Processing/SignalProcessor.wl"
<< "Mathematica/Optimization/GradientDescent.wl"

(* Generate and process signal *)
signal = GenerateSignal["Sine", {0, 2}, SamplingRate -> 1000, Frequency -> 50];
spectrum = ComputeFFT[signal, Window -> "Hann"];

(* Optimization example *)
rosenbrock[x_, y_] := 100*(y - x^2)^2 + (1 - x)^2;
result = GradientDescentMinimize[rosenbrock[x, y], {x, y}, {-1.2, 1.0}];

Print["Optimization result: ", result["Solution"]]
Print["Function value: ", result["FunctionValue"]]
```

---

## Performance Benchmarks

### Timing Comparisons

```python
import time
import numpy as np
from Python.Signal_Processing import SignalProcessor

def benchmark_fft_performance():
    """Benchmark FFT performance across different sizes."""
    
    processor = SignalProcessor(sampling_rate=1000)
    sizes = [1024, 2048, 4096, 8192, 16384]
    times = []
    
    for n in sizes:
        # Generate test signal
        signal = np.random.randn(n)
        
        # Time FFT computation
        start = time.time()
        for _ in range(100):  # Average over multiple runs
            freq, mag = processor.compute_fft(signal)
        elapsed = (time.time() - start) / 100
        times.append(elapsed)
        
        print(f"Size {n:5d}: {elapsed*1000:.3f} ms per FFT")
    
    # Theoretical O(N log N) scaling
    theoretical = [times[0] * (n/sizes[0]) * np.log2(n/sizes[0]) for n in sizes]
    
    print("\nScaling comparison:")
    for i, n in enumerate(sizes):
        ratio = times[i] / theoretical[i]
        print(f"Size {n:5d}: Actual/Theoretical = {ratio:.2f}")

benchmark_fft_performance()
```

This comprehensive guide demonstrates the power and flexibility of the Berkeley SciComp framework across multiple domains and platforms. Each example is designed to be educational while showcasing real-world applications in scientific computing, finance, and engineering.