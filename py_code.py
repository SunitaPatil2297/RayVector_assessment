
import numpy as np
import cv2 
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os
from tkinter import *
from tkinter import filedialog


class FindShape:
    def __init__(self, image_path):
        self.image_path = image_path
        self.shape_count = 0
        self.frame = self.image_frame_reading()
        self.thresh = self.image_preprocessing(self.frame)
        self.contours = self.finding_shape(self.thresh)
        # self.total_shapes = len(self.contours)

    
    def image_frame_reading(self):
        """Image Reading"""
        frame = cv2.imread(self.image_path)
        return frame
    
    def image_preprocessing(self, frame):
        """Image preprocessing:gray, threshold"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    
    def finding_shape(self, thresh):
        """Find Shape"""
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours
    
    def count_shapes(self):
        for contour in self.contours :
            epsilon = 0.03 * cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(self.frame, [contour], -1, (0, 255, 0), 2)
            # x,y,w, h = cv2.boundingRect( contour)
            # (center_x, center_y), (wi,hi), ang = cv2.minAreaRect( contour)
            # print(center_x, center_y)
            self.shape_count += 1
         
            
            # cv2.rectangle(self.frame, (x,y), (x+w, y+h), [0,255,0], 3)

            for i in range(len(corners)):
                x1, y1 = corners[i][0]
                print(x1, y1)
                x2, y2 = corners[(i + 1) % len(corners)][0]  
                print("x2,y2", x2,y2)
                
                cv2.circle(self.frame, (x1, y1), 3, (0, 0, 255), -1)
                
                distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
                print("dis", distance)
                
                if distance > 20:
                    
                    edge_length = int(distance)
                    
                    text_x = (x1 + x2) // 2
                    text_y = (y1 + y2) // 2
                    
                    cv2.putText(self.frame, str(edge_length), (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.frame, "NO OF Shapes "+str(self.shape_count), (20, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 124), 2)
        return self.frame, self.shape_count
    

def open_file():
    global filepath
    global shape_count
    
    filepath = askopenfilename(filetypes=[("All Files", "*"), ("All Files", "*.*")])
    if not filepath:
        return
    # return filepath
    image_path = filepath
    find_shape_obj = FindShape(image_path)
    frame, shape_count = find_shape_obj.count_shapes()
   
    # print(frame)
    
    cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

window = tk.Tk()
window.geometry("100x100")
window.title("RayVector assessment")

fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=1)
fr_buttons.place(x=5, y=5)

btn_open = tk.Button(fr_buttons, text="BROWSE", command=open_file)
btn_open.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# shape_count_label = tk.Label(fr_buttons, text={"Number of shapes: 0", shape_count})
# shape_count_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
window.mainloop()

window.destroy()