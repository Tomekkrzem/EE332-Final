import numpy as np
import cv2
from pathlib import Path

# Global Variables
GROUP_THRESH = 50
MIN_AREA = 2000
CIRCULARITY = 90
CALIBRATION_SQR_SIZE = 23.8


class ObjectMeasure:

    def __init__(self, UoM, eq):
        
        # If Unit of Measurement is 0 Configure to Inches 
        if UoM == 0:
            unit = "in"
            scale = 0.01229508196721 # Pixels-to-in
            rnd_num = 2

        # If Unit of Measurement is 1 Configure to Millimeters
        elif UoM == 1:
            unit = "mm"
            scale = 0.312295081967 # Pixels-to-mm
            rnd_num = 2

        # Else Leave as Pixel Count
        else:
            unit = "px"
            scale = 1
            rnd_num = 0

        # Scale Descriptor
        self.scale = [scale, unit, rnd_num]

        # Initialize an Image
        self.img = None

        # Equalization Method Specified by User
        self.eq_method = eq


    # STEP 1: Preprocessing and Edge Detection
    def preprocess(self):

        # Greyscale the Image
        g_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        if self.eq_method == 'Histogram':
            # Histogram Equalization
            equalized_img = cv2.equalizeHist(g_img)

        else: 
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=1.5)
            equalized_img = np.clip(clahe.apply(g_img), 0, 255).astype(np.uint8)

        # Gaussian Blur for Initial Blur
        blur_img = cv2.GaussianBlur(equalized_img, (9,9), 2)

        # Bilateral Filter Blur for Edge Emphasis
        blur_img = cv2.bilateralFilter(blur_img, 15, 50, 50)

        # Extract Upper and Lower Threshold via OTSU for Canny
        upper_thresh, _ = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_thresh = int(upper_thresh/2)

        # Perform Edge Detection Using Canny
        edge = cv2.Canny(blur_img, lower_thresh, upper_thresh)

        # Rectangle Structuring Element for Morphological Operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        # Morphological Operation: CLOSING (Connects Near Edges)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

        return edge


    # Function to Compute Centroids
    def centroid(self, contour):
        
        # Moments of the Contour
        M = cv2.moments(contour)

        # If the Object has an Area
        if M['m00'] != 0:
            
            # Compute the X and Y Components of the Centroid
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        return cx, cy


    # STEP 2: Geometric Properties
    def object_measure(self, contour):

        # Extract Centroid of Contour
        cx, cy = self.centroid(contour)

        # Compute Distance from Cetntroid to Perimeter Point
        dist = [np.hypot(pt[0][0] - cx, pt[0][1]- cy) for pt in contour]

        # Determine Longest and Shorteset Distance
        min_r, max_r = min(dist), max(dist)

        # Determine Ratio of Distance
        ratio = min_r / max_r

        # If Ratio is Near 1.0 (i.e. 0.9) Object is Circular
        if ratio > 0.9:

            # Extract Center Position and Radius of Bounding Circl
            (cir_x, cir_y), radius = cv2.minEnclosingCircle(contour)

            # Compute the Diameter of the Circle
            radius = int(radius)
            diameter = radius * 2 * self.scale[0]

            # Determine the Center of the Bounding Circle in Image Coordinates
            circ_cent = int(cir_x), int(cir_y)

            # Draw the Bounding Circle and Display its Corresponding Diameter
            cv2.circle(self.img, circ_cent, radius, (255, 16, 240), 2)
            cv2.putText(self.img, "D: " + str(round(diameter, self.scale[2])) + self.scale[1], (circ_cent[0] - 100, circ_cent[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 16, 240), 2)

        # Else Object is Not Circular
        else:   
            
            # If Object Area is Less than a Threshold Amount Omit It
            if cv2.contourArea(contour) < MIN_AREA:
                return

            # Extract the Bounding Box
            bound_box = cv2.minAreaRect(contour)

            # Determine the Center Position, Geometric Properties, and Angle of Rotation
            _, (bw, bh), _ = bound_box

            # Scale the Geometric Properties w.r.t Unit of Measurement
            width_meas = bw * self.scale[0]
            height_meas = bh * self.scale[0]

            # Determine a Bounding Box with OpenCV Function and Draw It
            bound_box_cnt = cv2.boxPoints(bound_box)
            cv2.drawContours(self.img, [bound_box_cnt.astype("int")], -1, (255, 16, 240), 2)

            # Determine Top Left Point of Bounding Box
            tl = min(bound_box_cnt, key=lambda p: (p[0] + p[1]))  

            # Get Coordinates of Top Left Point in Image Coordinates and Display the Width and Height
            x, y = int(tl[0]), int(tl[1]) 
            cv2.putText(self.img, "W: " + str(round(width_meas, self.scale[2])) + self.scale[1], (x - 80, y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 16, 240), 2)
            cv2.putText(self.img, "H: " + str(round(height_meas, self.scale[2])) + self.scale[1], (x - 80, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 16, 240), 2)


    # STEP 3: Offset Measurement
    def offset_measure(self, obj_centroid, curr_centroid, angle):

        # Extract Centroid of Internal Features
        cx, cy = curr_centroid          

        # Extract Centroid of Object
        cx_outer, cy_outer = obj_centroid   

        # Convert Angle to Radians
        theta = np.deg2rad(angle)

        # Compute Unit Vectors of the Object Frame
        horiz_x, horiz_y = np.cos(theta), np.sin(theta)                          

        # Compute the Offset Measurements in the Image Frame
        lx = cx - cx_outer
        ly = cy - cy_outer

        # Determine the Distance of the X-Offset Vector Project onto the X-Axis of the Body Frame
        dx = lx * horiz_x + ly * horiz_y  

        # Compute the Euclidean Distance between Object Centroid and Image Centroid
        dist = np.hypot(lx, ly)
        
        # Compute End Point of Line for Plotting the Rotationally Invariant Offset Vectors
        x_rot = cx_outer + dx * horiz_x
        y_rot = cy_outer + dx * horiz_y

        # Compute the Rotationally Invariant Offset Measurements
        distx = np.hypot(cx_outer - x_rot, cy_outer -y_rot)
        disty = np.hypot(x_rot - cx, y_rot - cy)

        # If the Scale is Pixel Count Set the Offset Distance of Perfectly Concentric Objects to Zero
        if self.scale[2] not in [1,2]:
            if distx <= 1 and disty <= 1:
                dist = 0

        # Convert All Points to Image Coordinates by making them Integer Values
        obj_centroid = (int(round(cx_outer)), int(round(cy_outer)))
        curr_centroid = (int(round(cx)), int(round(cy)))
        X_rot = (int(round(x_rot)), int(round(y_rot)))
        
        # PLot the Offset Vectors
        cv2.line(self.img, obj_centroid, curr_centroid, (0, 0, 255), 1)          # Distance Vector
        cv2.line(self.img, obj_centroid, X_rot, (255, 255, 0), 1)                # X-Offset Vector
        cv2.line(self.img, X_rot, curr_centroid, (0, 255, 255), 1)               # Y-Offset Vector

        # Plot the Offset Measurement Text
        sign_ly = np.sign(ly) if ly != 0 else 1
        cv2.putText(self.img, "D: " + str(round(self.scale[0] * dist, self.scale[2])) + self.scale[1], (cx - 20, cy + int(sign_ly * 60)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(self.img, "X: " + str(round(self.scale[0] * distx, self.scale[2])) + self.scale[1], (cx - 20, cy + int(sign_ly * 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(self.img, "Y: " + str(round(self.scale[0] * disty, self.scale[2])) + self.scale[1], (cx - 20, cy + int(sign_ly * 40)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    # Function that Performs all Measurements
    def measurements(self, img):
        
        # Assign Current Image
        self.img = img

        # Preprocess and Extract Edges
        edge = self.preprocess()

        # Determine Contours and their Hierarchy
        contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Extracts Root Contours from Hierarchy
        root_contours = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]

        # Group Objects if Root Contours are Very Close
        grouped_objects = []

        # Loop Through all Root Contours
        for r in root_contours:

            # Compute Centroid even if Area is near Zero
            M = cv2.moments(contours[r])
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))

            # Loop Through Grouped Objects
            for obj in grouped_objects:
                
                # If the Grouped Objects are Determined to be too Close Set the Root Contour of the Grouped Objects to the Current Contour
                if abs(cx-obj[0]) < GROUP_THRESH and abs(cy-obj[1]) < GROUP_THRESH: 
                    obj[2].append(r)
                    break  
            
            # Update Grouped Objects
            else: 
                grouped_objects.append([cx,cy,[r]])

        # Ensure Largest Contour is the Root Contour in Each Object
        root_contours = [max(obj[2], key=lambda i: cv2.contourArea(contours[i])) for obj in grouped_objects]

        # Get the Inside Contour of the Root Contour
        inner_roots = [i for r in root_contours for i in range(len(contours)) if hierarchy[0][i][3] == r]

        # Initialize Root Centroid
        root_center = None

        # Loop Through all Detected Contours
        for i, cnt in enumerate(contours):
            
            # If the Contour is a Root Contour
            if i in inner_roots:
                
                # Extract its Centroid
                cx, cy = self.centroid(cnt)

                # Update Root Centroid and Root Contour
                root_center = cx, cy
                root_cnt = cnt

                # Draw a Red Circle for the Centroid
                cv2.circle(self.img, (cx, cy), 2, (0, 0, 255), -1)

                # Draw a Green Contour
                cv2.drawContours(self.img, contours, i, (0, 255, 0), 2)

                # Get Geometric Properties
                self.object_measure(cnt)

            # If the Contour is an Internal Feature Contour
            else:
                
                # If the Area is Positive (Inside Contour)
                if cv2.contourArea(cnt, True) > 0:
                    
                    # Extract its Centroid
                    cx, cy = self.centroid(cnt)

                    # Draw a Red Circle for the Centroid
                    cv2.circle(self.img, (cx, cy), 2, (0, 0, 255), -1)

                    # Draw a Green Contour
                    cv2.drawContours(self.img, contours, i, (0, 255, 0), 2)

                    # If a Root Centroid Exists
                    if root_center:
                        
                        # Extract a Centroid and Angle of Rotation for Outer Contour
                        rect = cv2.minAreaRect(root_cnt)
                        (center, _, angle) = rect

                        # Get Offset Measurements
                        self.offset_measure(center, (cx,cy), angle)

        return self.img, edge



class Camera:

    def __init__(self, unit_of_measure, equalization_method):
        
        # Initialize Measurements
        self.Measure = ObjectMeasure(unit_of_measure, equalization_method)
    
    def image(self, img_path):

        # Get Image and Resize It
        img = cv2.imread(img_path)
        img = cv2.resize(img, (720, 720))

        # Perform Measurements and Get Edge Detection
        img, edge = self.Measure.measurements(img)

        # Show the Measurements on the Image and the Edge Detection
        cv2.imshow("Image", img)	
        cv2.imshow("Edge", edge)	

        # When Closed Kill all Windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def video(self):
        
        # Stream the Video from the Camera
        stream = cv2.VideoCapture(0)

        while True:
            
            # Exract the Frame
            _, frame = stream.read()
            
            # Get Measurements and Edge Detection of Current Frame
            frame, edge = self.Measure.measurements(frame)

            # Resize the Frame
            frame = cv2.resize(frame, (1080, 720))

            # Show the Measurements on the Image and the Edge Detection
            cv2.imshow('Video Stream', frame)
            cv2.imshow('Edge Stream', edge)
            
            # If Q is Pressed End the Video Stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # Test Images
    img_path1 = Path(__file__).parent / "Test Image.png"
    img_path2 = Path(__file__).parent / "Test Image 2.png"
    img_path3 = Path(__file__).parent / "Test Image 3.png"
    img_path4 = Path(__file__).parent / "Test Image 4.png"

    # Testing the Test Images
    I = Camera(2, "Histogram")
    I.image(img_path1)
    I.image(img_path2)
    I.image(img_path3)
    I.image(img_path4)
    
    # Video Stream
    V = Camera(0, "CLAHE")
    V.video()