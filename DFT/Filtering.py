import numpy as np
import math


class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order=0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order

    def get_ideal_low_pass_filter(self, shape, cutoff):
        mask = np.zeros((shape[0], shape[1]))
        for u in range(shape[0]):
            for v in range(shape[1]):
                d = np.sqrt((u - (shape[0] / 2)) ** 2 + (v - (shape[1] / 2)) ** 2)
                if d <= cutoff:
                    mask[u, v] = 1
                else:
                    mask[u, v] = 0

        return mask

    def get_ideal_high_pass_filter(self, shape, cutoff):


        # Hint: May be one can use the low pass filter function to get a high pass mask
        low_pass = self.get_ideal_low_pass_filter(shape, cutoff)
        mask = 1 - low_pass

        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):

        mask = np.zeros((shape[0], shape[1]))
        for u in range(shape[0]):
            for v in range(shape[1]):
                d = np.sqrt((u - (shape[0] / 2)) ** 2 + (v - (shape[1] / 2)) ** 2)
                mask[u, v] = 1 / ((1 + d / cutoff) ** (2 * order))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):

        # Hint: May be one can use the low pass filter function to get a high pass mask
        low_pass = self.get_butterworth_low_pass_filter(shape, cutoff, order * -1)
        mask = low_pass

        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):

        mask = np.zeros((shape[0], shape[1]))
        for u in range(shape[0]):
            for v in range(shape[1]):
                d = np.sqrt((u - (shape[0] / 2)) ** 2 + (v - (shape[1] / 2)) ** 2)
                mask[u, v] = np.exp((-d ** 2) / (2 * cutoff ** 2))

        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):


        # Hint: May be one can use the low pass filter function to get a high pass mask
        low_pass = self.get_gaussian_low_pass_filter(shape, cutoff)
        mask = 1 - low_pass

        return mask

    def post_process_image(self, image):
        a = np.min(image)
        b = np.max(image)
        p = (255 - 1) / (b - a)
        post_proc_image = np.zeros((image.shape[0], image.shape[1]))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                post_proc_image[i, j] = p * (image[i, j] - a)

        return post_proc_image

    def filtering(self):

        # image = cv2.imread('Lenna.png', 0)
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        magnitude_dft = np.abs(fshift)
        magnitude_dft[magnitude_dft == 0] = 1
        magnitude_dft = np.log(magnitude_dft).astype('uint8')
        magnitude_dft = self.post_process_image(magnitude_dft)
        if self.order != 0:
            mask = self.filter(self.image.shape, self.cutoff, self.order)
        else:
            mask = self.filter(self.image.shape, self.cutoff)
        mask = np.round(mask)
        product = mask * fshift
        magnitude_idft = np.abs(product)
        magnitude_idft[magnitude_idft == 0] = 1
        magnitude_idft = np.log(magnitude_idft).astype('uint8')
        magnitude_idft = self.post_process_image(magnitude_idft)
        filtered_image = np.fft.ifftshift(product)
        ifft = np.fft.ifft2(filtered_image)
        magnitude_ifft = np.abs(ifft)
        filtered_image = self.post_process_image(magnitude_ifft)

        return [filtered_image, magnitude_dft, magnitude_idft]
