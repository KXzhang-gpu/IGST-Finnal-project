# IGST-Finnal-project
This is code for Image Guided Surgery and Therapy finnal project

## Requirements:
- PyQt == 5.15
- numpy >= 1.26
- opencv-python
- matplotlib

## Project details
### Project-1 
- Histogram and threshold
- Requirements：
  - Realize Histogram analysis and threshold operation
  - threshold operation can be mannual or automatic(such as Ostu and Entropy)
- The code of project-1 is here : [threshold.py](threshold.py)

### Project-2
- Convolution and Image Filters
- Requirements:
  - Realize the convolution operation
  - Realize Roberts operator, Prewitt operator, Sobel operator
  - Realize Gaussian filter and Median filter
  - Using the operator and filter to realize edge detection and noise reduction
- The code of project-2 is here : [convolution.py](convolution.py)
  - There are 3 ways to implement convolution operation:
    1. Implementation according to defination: Calculate the convolution with the kernel at each position, as seen at the function conv().
    2. Img2Col convolution acceleration algorithm: Convert the image into a vector and then calculate the results all at once, you can find it at the function conv2d_img2col()
    3. Use FFT to convert spatial domain convolution into frequency domain multiplication, enabling direct calculation. Then, use iFFT to obtain the tracking result, implementation at function conv2d_fft()

### Project-3
- sample Morphology algorithms
- Requirements:
  - Realize binary erosion, dilation, opening, closing
  - Try to apply fast operations in case
- The code of project-3 is here : [morphology.py](mophology.py)

### Project-4
- complex Morphology algorithms
- Requirements:
  - Realize morphological distance transform
  - Realize morphological skeleton
  - Realize morphological skeleton restoration
- The code of project-4 is here : [morphology.py](morphology.py)

### Project-5
- complex Morphology algorithms
- Requirements:
  - Realize gray scale erosion, dilation, opening, closing
- The code of project-5 is here : [morphology.py](morphology.py)

### Project-6
- complex Morphology algorithms
- Requirements:
  - Realize morphological edge detection and gradient
  - Realize conditional dilation of binary image
  - Realize gray scale reconstruction
- The code of project-6 is here : [morphology.py](morphology.py)
