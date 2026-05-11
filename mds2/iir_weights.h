#!/usr/bin/env python3
from scipy import signal

# --- Filter Specifications ---
fs = 20000   # Sampling rate (Hz)
fc = 300     # High-pass cutoff frequency (Hz)
order = 2    # 2nd-order Biquad filter

# Generate IIR Butterworth coefficients
b, a = signal.butter(order, fc, btype='highpass', fs=fs)

# --- Define Output Filename ---
output_filename = "iir_weights.h"

# --- Format the C Header File ---
# Note: We use {{ and }} to escape literal curly braces in Python f-strings
c_header_content = f"""// Auto-generated IIR Biquad Coefficients
// Type: High-pass Butterworth
// Order: {order}
// Sampling Rate: {fs} Hz
// Cutoff Frequency: {fc} Hz

#ifndef IIR_WEIGHTS_H
#define IIR_WEIGHTS_H

// Biquad Coefficient Array
// Index map: [0]=b0, [1]=b1, [2]=b2, [3]=a1, [4]=a2
// Note: a0 is mathematically always 1.0, so it is omitted.
const float iir_weights[5] = {{
    {b[0]:.8f}f, 
    {b[1]:.8f}f, 
    {b[2]:.8f}f, 
    {a[1]:.8f}f, 
    {a[2]:.8f}f
}};

#endif // IIR_WEIGHTS_H
"""

# --- Write to File ---
with open(output_filename, "w") as f:
    f.write(c_header_content)

print(f"Successfully generated {output_filename} as an array.")
