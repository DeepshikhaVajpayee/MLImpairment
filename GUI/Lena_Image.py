import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load both images
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Convert BGR to RGB for visualization
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Define 10 filters manually
def apply_filters(image):
    filters = []
    filters.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))  # 1. Grayscale
    filters.append(cv2.Canny(image, 100, 200))              # 2. Edge detection
    filters.append(cv2.GaussianBlur(image, (7,7), 0))       # 3. Gaussian blur
    filters.append(cv2.medianBlur(image, 7))                # 4. Median blur
    filters.append(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F))  # 5. Laplacian
    filters.append(cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=5)) # 6. Sobel X
    filters.append(cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=5)) # 7. Sobel Y
    filters.append(cv2.bilateralFilter(image, 9, 75, 75))   # 8. Bilateral filter
    filters.append(cv2.erode(image, np.ones((5,5), np.uint8), iterations=1))  # 9. Erosion
    filters.append(cv2.dilate(image, np.ones((5,5), np.uint8), iterations=1)) # 10. Dilation
    return filters

filters_img1 = apply_filters(img1)
filters_img2 = apply_filters(img2)

# Function to plot histograms
def plot_histogram(image, title):
    plt.figure(figsize=(5,4))
    if len(image.shape) == 2:  # Grayscale
        plt.hist(image.ravel(), 256, [0,256], color='black')
    else:  # Color image
        color = ('r','g','b')
        for i,col in enumerate(color):
            hist = cv2.calcHist([image],[i],None,[256],[0,256])
            plt.plot(hist,color=col)
    plt.title(f'Histogram - {title}')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

# Show filtered images and histograms for comparison
for i, (f1, f2) in enumerate(zip(filters_img1, filters_img2)):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    cmap1 = 'gray' if len(f1.shape)==2 else None
    plt.imshow(f1, cmap=cmap1)
    plt.title(f'Image 1 - Filter {i+1}')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    cmap2 = 'gray' if len(f2.shape)==2 else None
    plt.imshow(f2, cmap=cmap2)
    plt.title(f'Image 2 - Filter {i+1}')
    plt.axis('off')
    plt.show()
    
    # Show histograms
    plot_histogram(f1, f'Image 1 - Filter {i+1}')
    plot_histogram(f2, f'Image 2 - Filter {i+1}')
