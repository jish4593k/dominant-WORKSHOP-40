import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import utils
import cv2

# Function to perform K-Means clustering and display the results
def perform_kmeans():
    image_path = image_path_var.get()
    num_clusters = int(clusters_var.get())

    if not image_path:
        messagebox.showerror("Error", "Please select an image.")
        return

    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.axis("off")
        plt.imshow(image)

        # Reshape the image
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # Cluster the pixel intensities
        clt = KMeans(n_clusters=num_clusters)
        clt.fit(image)

        # Build a histogram of clusters and create a figure
        hist = utils.centroid_histogram(clt)
        bar = utils.plot_colors(hist, clt.cluster_centers_)

        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create a Tkinter window
window = tk.Tk()
window.title("Color K-Means Clustering")
window.geometry("400x200")

# Create and pack a label for instructions
label = tk.Label(window, text="Select an image and number of clusters:")
label.pack(pady=10)

# Create a button to open the file dialog
image_path_var = tk.StringVar()
open_button = tk.Button(window, text="Open Image", command=lambda: browse_file())
open_button.pack()

# Create a dropdown for selecting the number of clusters
clusters_var = tk.StringVar()
clusters_label = tk.Label(window, text="Number of Clusters:")
clusters_label.pack()
clusters_dropdown = ttk.Combobox(window, textvariable=clusters_var, values=[2, 3, 4, 5, 6, 7, 8, 9, 10])
clusters_dropdown.set(3)  # Default value
clusters_dropdown.pack()

# Create a button to perform K-Means clustering
calculate_button = tk.Button(window, text="Perform K-Means Clustering", command=perform_kmeans)
calculate_button.pack(pady=10)

# Function to browse and select an image file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        image_path_var.set(file_path)

# Start the Tkinter main loop
window.mainloop()
