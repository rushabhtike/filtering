# Report
# Rushabh Tike 
# PSID: 1800104

Part 1: DFT

a. Forward Transform
- Used the formula discussed in class to calculate forward transform for the given matrix.
- Used 4 for loops to traverse the matrix and compute the fft.
- Used cmath library to calculate value of the exponential.
- Formula: 
    - x = -1 * cmath.sqrt(-1) * (2 * math.pi * (u * i + v * j)) / N
    - val = val + (matrix[i, j] * cmath.exp(x))

b. Inverse Transform
- Used the formula discussed in class to implement inverse transform.
- Used np.zeros to create an empty array to store the result.
- Formula:
    - x = cmath.sqrt(-1) * (2 * math.pi * (u * i + v * j)) / N
    - val = val + (1 / (N * N)) * (matrix[i, j] * cmath.exp(x))
    
c. Discrete Cosine Transform
- Used the function discussed in class to implement dct.
- Formula:
    - x = (2 * math.pi * (u * i + v * j)) / 15
    - val = val + (matrix[i, j] * math.cos(x))
    
d. Magnitude
- Formula: 
    - val = cmath.sqrt((matrix[u, v].real * matrix[u, v].real) + (matrix[u, v].imag * matrix[u, v].imag))
    - mag[u, v] = val
    
    
Part 2: Filtering

a. Filtering
- Computed the fft of image by using np.fft.fft2
- Computed the fftshift by using np.fft.fftshift
- Got the magnitude,did logarithmic compression and converted to unit8 to save an image that is visible.
- Did a full contrast stretch in post processing.
- Applied the filters using self.filter() and convolution theorem.
- Computed inverse shift  and the inverse fourier transform.
- Computed the magnitude and did logarithmic compression.
- Did the post processing, ifft shift, ifft.
- Got the magnitude by using np.abs().
- Returned the filtered image, magnitude_dft and magnitude_idft.

b. Ideal Low Pass Filter
- Calculated the distance by using formula:
    - d = np.sqrt((u - (shape[0] / 2)) ** 2 + (v - (shape[1] / 2)) ** 2)
- Got the filter by comparing distance with cutoff value.

c. Ideal High Pass Filter
- Got the high pass filter by doing 1 - low_pass_mask

d. Butterworth Low Pass Filter
- Got the filter by using formula:
    - mask[u, v] = 1 / ((1 + d / cutoff) ** (2 * order))
    
e. Butterworth High Pass Filter
- Got the filter by using the function for low pass and multiplying the order by -1

f. Gaussian Low Pass Filter
- Got the filter by using the formula:
    - mask[u, v] = np.exp((-d ** 2) / (2 * cutoff ** 2))

g. Gaussian High Pass Filter
- Got the filter by using by doing 1 - low_pass_mask


h. Post Processing
- Did a full contrast stretch in post processing.
- Got the min and max values by using np.min() and np.max()
- Used the following formulas to do full contrast stretch:
    - p = (255 - 1) / (b - a)
    - post_proc_image[i, j] = p * (image[i, j] - a)

