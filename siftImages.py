
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys


'** TASK 1 **'


# VGA Size (480(rows) x 600(columns)) as Mentioned in the Question

# Function to resize an image to VGA resolution while maintaining the aspect ratio

def image_resize(image_path):
    
    # Loading the image

    image = cv.imread(image_path)

    # Error handling if the image is not found

    if image is None:

        print("\nImage file not found. Please check the file path.\n")

        sys.exit(1)
   
    else:
        # Getting the original dimensions of the inputted image

        height, width = image.shape[:2]

        # Defining the VGA resolution

        width_vga = 600
        height_vga = 480

        # Calculating the scaling factors for width and height

        width_scale = width_vga / width
        height_scale = height_vga / height

        # Choose the smaller scaling factor to maintain aspect ratio

        scale = min(width_scale, height_scale)

        # Calculating the new dimensions after blur downsize

        new_width = int(width * scale)
        new_height = int(height * scale)


        # Resizing the image to VGA size

        resized_image = cv.resize(image, (new_width, new_height),interpolation=cv.INTER_LINEAR)

        # Changing the color space of the resized image to XYZ

        xyz_format = cv.cvtColor(resized_image, cv.COLOR_BGR2XYZ)

        # Extracting the Y channel from the Image

        x, y, z = cv.split(xyz_format)
        y_channel = y

        # Returning both the Y channel and the resized image

        return y_channel, resized_image

# Function to perform SIFT keypoint detection

def sift_detection(image_path):

    # Creating a SIFT detector

    sift = cv.SIFT_create()

    # Resizing the image to VGA and obtain the Y channel

    y_channel, _ = image_resize(image_path)

    # Detecting keypoints and compute descriptors

    keypoints, descriptors = sift.detectAndCompute(y_channel, None)

    # Returning identified keypoints and descriptors

    return keypoints, descriptors


# Function to draw keypoints on an image

def draw_keypoints(image, keypoints):

    img_copy = image.copy()

    # For loop to go throught all the identified keypoints

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size = int(keypoint.size)
        angle = keypoint.angle

        # Calculating the end of the orientation line

        end_x = int(x + size * np.cos(np.deg2rad(angle)))
        end_y = int(y + size * np.sin(np.deg2rad(angle)))

        # Drawing a circle around the keypoint

        cv.circle(img_copy, (x, y), size, (0, 255, 0), 1)

        # Drawing the orientation line

        cv.line(img_copy, (x, y), (end_x, end_y), (0, 255, 0), 1)

        # Draw the cross at the center

        cross_size = 2
        cv.line(img_copy, (x, y - cross_size), (x, y + cross_size), (0, 0, 255), 1)
        cv.line(img_copy, (x - cross_size, y), (x + cross_size, y), (0, 0, 255), 1)

    return img_copy


'** TASK 2 **'

  # Function to compute the histogram of descriptors

def descriptor_histogram(descriptors, centers):
        
        k = centers.shape[0]
        histogram = np.zeros(k)
        for descriptor in descriptors:
            id = np.argmin(np.linalg.norm(descriptor - centers, axis=1))
            histogram[id] += 1
        return histogram
    
    # Function to compute the distance of histograms 

def histogram_distance(histA, histB, eps=1e-10):

        # Ensure that the histograms are numpy arrays

        histA = np.array(histA)
        histB = np.array(histB)

        # Normalize both histograms

        histA_normalized = histA / (np.sum(histA) + 1e-10)
        histB_normalized = histB / (np.sum(histB) + 1e-10)

        # Calculate the χ² 
        
        diff = 0.5 * np.sum(((histA_normalized - histB_normalized) ** 2) / (histA_normalized + histB_normalized + 1e-10))

        return diff

# Function to compare images using SIFT Algorithm

def sift_comparison(image_files):

    # Initializing variables 

    all_descriptors = []
    image_features = {}
    total_keypoints = 0
            
    # For loop to go through image files

    for image_path in image_files:

        # Detecting keypoints and compute descriptors

        keypoints, descriptors = sift_detection(image_path)

        # Printing the number of keypoints for each image

        print("# of keypoints in {} is {}".format(image_path, len(keypoints)))
        total_keypoints += len(keypoints)

        # Storing descriptors and keypoints for each image

        if descriptors is not None:

            all_descriptors.append(descriptors)

            image_features[image_path] = {'keypoints': keypoints, 'descriptors': descriptors}

    # If Keypoints identified

    if total_keypoints > 0:

        print("Total number of keypoints for all images: {}\n".format(total_keypoints))

    # If no keypoints identified

    else:

        print("No keypoints found in any of the images.")

        return

    # If descriptors available

    if all_descriptors:

        all_descriptors = np.vstack(all_descriptors).astype(np.float32)

        # Specifying K-Means percentages

        K_percentages = [0.05, 0.10, 0.20]

        # For Loop to run through each specified percentages

        for K_percentage in K_percentages:
            
            K = int(K_percentage * total_keypoints)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)

            # K-means clustering on the descriptors

            kmeans_result = cv.kmeans(all_descriptors, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
            centers = kmeans_result[2]

            image_histograms = {}

            # Creating descriptor histogram for each image

            for image_path, features in image_features.items():

                image_histograms[image_path] = descriptor_histogram(features['descriptors'], centers)

            distances = {}

            # Calculating dissimilarity distances

            for imageA_path in image_files:

                distances[imageA_path] = {}

                for imageB_path in image_files:

                    if imageA_path == imageB_path:

                        distances[imageA_path][imageB_path] = 0.0

                    else:
                        distances[imageA_path][imageB_path] = histogram_distance(
                            image_histograms[imageA_path], image_histograms[imageB_path])

           # Printing the output

            print("\nK={}% * total keypoints = {}\n".format(K_percentage * 100, K))
            print("Dissimilarity Matrix\n")

            # Calculating the maximum length of the image filenames for alignment

            max_length = max([len(image_path) for image_path in image_files])

            # Calculate the width of each column (including padding)

            column_width = max_length + 2 

            # Printing the header 

            header_format = "{{:^{}}}".format(column_width)
            print(header_format.format(""), end="")  

            for imageB_path in image_files:

                print(header_format.format(imageB_path), end="")

            print()  

            # Printing the matrix

            for i, imageA_path in enumerate(image_files):

                print(header_format.format(imageA_path), end="")  

                for j, imageB_path in enumerate(image_files):

                    if j >= i:
                        distance = distances[imageA_path][imageB_path]
                        distance_format = "{{:^{}}}".format(column_width)
                        print(distance_format.format(f"{distance:.4f}"), end="")

                    else:

                        # Printing a blank space for the lower diagonal

                        print(header_format.format(""), end="")

                # New line after each row
                print() 

            print()


# Main function
def main():
    image_files = sys.argv[1:]

    # IF image_files length is 1 sift_detection Function is Triggered

    if len(image_files) == 1:
        
        if image_files is not None:

            # Loading the  image

            image_path = image_files[0]
            y_channel, resized_img = image_resize(image_path)
            keypoints, des = sift_detection(image_path)

            keypoint_img = draw_keypoints(resized_img, keypoints)

            # Horizontally stacking the two images

            stacked_img = np.hstack((resized_img, keypoint_img))

            # Printing Image Size

            h,w = resized_img.shape[:2]

            print("Image Dimentions (After Downsizing with Aspect Ratio) : {} Rows x {} Columns ".format(h,w))

            # Output Number of Keypoints 
            
            print("# of keypoints in {} is {}".format(image_path, len(keypoints)))

            # Calculating the dimensions of the stacked image

            stacked_height, stacked_width, _ = stacked_img.shape

            # Creating a window to display the stacked images

            cv.namedWindow("Original Image (Rescaled) and Image with Key Points Detected", cv.WINDOW_NORMAL)

            # Setting the custom window size 

            cv.resizeWindow("Original Image (Rescaled) and Image with Key Points Detected", stacked_width, stacked_height)

            # Show the stacked images in the window

            cv.imshow("Original Image (Rescaled) and Image with Key Points Detected", stacked_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        else:

            print("Image not found")


    # IF image_files length is more than 1 sift_comparison Function is Triggered        

    elif len(image_files) > 1:

        sift_comparison(image_files)

    # IF no image_files are passed   

    else:
        print("\nUsage: python siftImages.py <imagefile1> (SIFT Keypoint Detection) or <imagefile1 imagefile2,imagefile3,…> (SIFT Image Comparison)\n")


if __name__ == "__main__":
    main()
