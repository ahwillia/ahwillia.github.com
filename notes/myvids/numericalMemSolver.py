#!/usr/bin/env ipython

import sys
import numpy as np
import pylab as sp

def specifyCurrent(tDel,tMax,dt,currType,currAmp):
	"""Constructs I_inj, the vector/array specifying the injected
	current.
	
	Inputs:
		tDel:     [float] the delay (in ms) before current begins
		            to be injected.
		tMax:     [float] the length of the simulation (in ms)
		dt:       [float] the time between samples. That is, the 
		            integration time step.
		currType: [string] Specifies the kind of current injection
		            protocol we are using.
		currAmp:  [float] Scales the amplitude of the current injection.
		            Larger amplitudes 
		
	Returns:
		I_inj: [nx1 array/vector] holds the amount of injected current
		         in nano-Amps at each time point.
	"""
	
	## Construct the current injection (part after the 'delay')
	t_curr = np.arange(0,tMax-tDel,dt) # time pts for current inj
	if currType == 'currentStep':
		activeCurr = np.ones(len(t_curr)) * currAmp
	elif currType == 'sine':
		f = 80/1e3   # 10 Hz sine wave
		activeCurr = currAmp * 0.5 * np.sin(2*np.pi*f*t_curr)
	elif currType == 'white':
		wn = np.random.randn(len(t_curr))
		wn = currAmp*(wn/(np.max(wn) - np.min(wn)))
		activeCurr = wn
	elif currType == 'pink':
		# Note: this isn't strictly pink noise. True pink noise would
		# divide the white noise FFT by the sqrt of the frequency. But 
		# scientists often use the term 'pink noise' loosely anyway (as
		# far as I can tell).
		t = np.arange(dt,tMax-tDel,dt)
		n = len(t_curr)
		wn = np.random.randn(n)     # white noise trace
		wn_F = np.fft.fft(wn,n)     # Fourier Transform of white noise
		freq = np.fft.fftfreq(n)    # Frequencies for wn_F
		freq[0] = 1                 # Avoid divide by zero
		pn_F = wn_F/(abs(freq)**0.7) # Pink noise has 1/F power spec.
		pn_F[0] = 0                  # ignore divide by zero
		pn = np.real(np.fft.ifft(pn_F))
		pn = currAmp*(pn/(np.max(pn) - np.min(pn)))
		activeCurr = pn

	## Put the current injection in after the initial "delay"
	I_inj = np.zeros(round(tMax/dt)) # vector of injected current
	start = len(I_inj)-len(t_curr)
	I_inj[start:] = activeCurr
	return I_inj

def solveMembraneNumerical(I_inj,R,t):
	"""Numerically solves for V(t) using the numerical procedure
	outlined in Dayan and Abbott's "Theoretical Neuroscience".
	
	Inputs:
		I_inj: [nx1 array/vector] holds the amount of injected current
		         at each time point in the simulation. [nA]
		R:     [float] specifies the membrane resistance in Megohms
		t:     [nx1 array/vector] holds the time of each sample in the
		         simulation. For example, if the time step ("dt") was
		         equal to 0.02, and the length of the simulation was 3,
		         then t = [0, 0.02, 0.04, ... , 3.96, 3.98].
	Returns:
		V:     [nx1 array/vector] holds the membrane potential at each
		         time point or sample.
	"""
	E = -60      # Resting membrane potential (-60 mV)
	C = 1        # Membrane Capacitance in nF
	V = np.zeros(len(t))
	V[0] = E     # Set initial condition to resting membrane potential
	tau = R*C
	dt = t[1]-t[0]
	for i in range(0,len(V)-1):
		V_ss = E + I_inj[i]*R
		V[i+1] = V_ss - (V_ss - V[i])*np.exp(-dt/tau)
	return V

if __name__ == '__main__':
	"""Numerically solve V(t) for a passive membrane
	
	Arugments:
		1) Integer number 1-4 specifying the type of current injection
			{1 = 'current step'; 2 = 'sine wave'; 3 = 'white noise';
			 4 = 'pink noise'}... Default is 'pink noise'
		2) Float corresponding to the amplitude of the current injection
		    (in nano-Amps)... Default is 5.0
		3) Float corresponding to the input resistance (R, in meg-ohms)
		    ... Default is 3.0 
	
	Other parameters (e.g. the frequency of the sine wave) cannot be
	changed from the command-line. You will need to alter the code
	itself. (Sorry!)
	
	Examples (from Bash command-line)
	---------------------------------
	Run simulation with default parameters
		./numericalMemSolver.py
	A current step of 3 nA, with a input resistance of 10 MOhms
		./numericalMemSolver.py 1 3 10  
	A sinusoidal wave with an amplitude of 6 nA, R = 8 MOhms
		./numericalMemSolver.py 2 6 8  
	White noise with an amplitude of 30 nA, R = 10 MOhms
		./numericalMemSolver.py 3 30 10 
	"""
	tDel = 0.02  # Delay before injecting current (ms)
	tMax = 5e2  # Simulation Length (ms)
	dt = 0.02   # Time step of integration
	
	# Parse inputs
	if len(sys.argv) == 4:
		currAmp = float(sys.argv[2])
		R = float(sys.argv[3]); # input resistance [Megohms]
		if sys.argv[1] == '1':
			currType = 'currentStep'
			tMax = 1e2  # More informative to zoom in on a current step
			tDel = 10
		elif sys.argv[1] == '2':
			currType = 'sine'
		elif sys.argv[1] == '3':
			currType = 'white'
		else:
			currType = 'pink'
	else:
		print 'Inputs to numericalMemSolver are not correct...'
		print 'Resorting to default parameters...'
		currType = 'pink'
		currAmp = 5    # [units = nA]
		R = 3;       # input resistance [units = Megohms]
	
	# Construct injected current and then solve for V(t)
	I_inj = specifyCurrent(tDel,tMax,dt,currType,currAmp)
	t = np.arange(0,tMax,dt)
	V = solveMembraneNumerical(I_inj,R,t)
	
	# Plot Results
	sp.figure()
	sp.subplot(2,2,1)
	sp.plot(t/1e3,V,'-b')
	Vrange = np.max(V)-np.min(V)
	sp.ylim([np.min(V)-(Vrange/2), np.max(V)+(Vrange/2)])
	sp.ylabel('Membrane Potential (mV)')
	
	sp.subplot(2,2,2)
	Vf = np.fft.fftshift(abs(np.fft.fft(V)**2)) # Fourier Transform
	Vf = Vf[(1+len(Vf)/2):]                     # Frequencies > 0
	sp.plot(Vf[1:100],'-b')
	sp.ylabel('Power - Membrane Potential')
	
	sp.subplot(2,2,3)
	sp.plot(t/1e3,I_inj,'-r')
	Irange = np.max(I_inj)-np.min(I_inj)
	sp.ylim([np.min(I_inj)-(Irange/2), np.max(I_inj)+(Irange/2)])
	sp.ylabel('Injected Current (nA)')
	sp.xlabel('time (s)')
	
	sp.subplot(2,2,4)
	If = np.fft.fftshift(abs(np.fft.fft(I_inj)**2)) # Fourier Transform
	If = If[(1+len(If)/2):]                         # Frequencies > 0
	sp.plot(If[1:100],'-r')
	sp.ylabel('Power - Injected Current')
	sp.xlabel('Frequencies')
	sp.show()
