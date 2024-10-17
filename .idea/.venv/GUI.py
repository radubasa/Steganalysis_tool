import tkinter as tk
from tkinter import messagebox
from tkinter import ttk 
from tkinter import filedialog
from PIL import Image, ImageTk
import steganalysis
import numpy as np
from scipy.stats import skew, kurtosis  # Add this line
from tkinter import simpledialog  # Add this line
import joblib  # Add this line

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("My Application")
        self.geometry("900x600")

        style = ttk.Style()  
        style.theme_use('clam') 

        self.image_label = ttk.Label(self)  # Add this line
        self.image_label.grid(row=2, column=0, columnspan=4)  # Add this line

        self.text_area = tk.Text(self)  # Initialize self.text_area
        self.text_area.grid(row=3, column=0, columnspan=4, sticky="nsew")

        self.tree = ttk.Treeview(self)  # Initialize self.tree
        self.tree["columns"] = ("one", "two")  # Add this line
        self.tree.column("#0", width=270, minwidth=270, stretch=tk.NO)  # Add this line
        self.tree.column("one", width=150, minwidth=150, stretch=tk.NO)  # Add this line
        self.tree.column("two", width=400, minwidth=200)  # Add this line
        self.tree.heading("#0", text="Name", anchor=tk.W)  # Add this line
        self.tree.heading("one", text="Value", anchor=tk.W)  # Add this line
        self.tree.grid(row=3, column=0, columnspan=4, sticky="nsew")  # Add this line



        ttk.Label(self, text="Hello to my application!").grid(row=0, column=0, columnspan=4)

        ttk.Button(self, text="Email Functionality", command=self.email_functionality).grid(row=1, column=0, padx=10, pady=10, sticky="ew", ipady=10)  # Modify this line
        ttk.Button(self, text="Info", command=self.show_info).grid(row=1, column=1, padx=10, pady=10, sticky="ew", ipady=10)  # Modify this line
        ttk.Button(self, text="Separate Functionalities", command=self.separate_functionalities).grid(row=1, column=2, padx=10, pady=10, sticky="ew", ipady=10)  # Modify this line
        ttk.Button(self, text="Analyze Image", command=self.analyze_image).grid(row=1, column=3, padx=10, pady=10, sticky="ew", ipady=10)  # Modify this line

    def email_functionality(self):
        # Implement your email functionality here
        print("Email Functionality")

    def show_info(self):
        messagebox.showinfo("Info", "This is some information about the application.")

    def separate_functionalities(self):
        # Implement your separate functionalities here
        print("Separate Functionalities")

    def analyze_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if image_path:
            print(f"Image path: {image_path}")

        # Open the image file
        img = Image.open(image_path)

        # Resize the image to fit the window
        img = img.resize((700, 400), Image.LANCZOS)

        # Save the resized image
        img.save("resized_image.jpg")

        # Convert the image to a PhotoImage
        photo = ImageTk.PhotoImage(img)

        # Display the image in the label
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference to the image to prevent it from being garbage collected

        # Convert the image to grayscale and to a numpy array
        image_matrix = np.array(img.convert("L"))

        # Ask the user for their choice
        choice = messagebox.askquestion("Choose analysis type", "Do you want to perform statistical analysis? If you choose No, ML analysis will be performed.", icon='warning')
        if choice == 'yes':
            # Call the functions for RS analysis, LBP, Chi Square attack, and Sample Pair Analysis
            lbp = steganalysis.calculate_lbp(image_matrix)  # Modify this line
            chi_square_stat = steganalysis.chi_square_attack(image_matrix)  # Modify this line
            spa_stat = steganalysis.sample_pair_analysis(image_matrix)  # Modify this line

    #       # Calculate the relevant statistics
            mean_lbp = np.mean(lbp)
            std_lbp = np.std(lbp)
            skew_lbp = skew(lbp.flatten())
            kurtosis_lbp = kurtosis(lbp.flatten())

            # Print the results and the statistics in the table
            self.tree.insert("", "end", text="LBP", values=(lbp,))  # Modify this line
            self.tree.insert("", "end", text="Mean LBP", values=(mean_lbp,))  # Add this line
            self.tree.insert("", "end", text="Standard deviation of LBP", values=(std_lbp,))  # Add this line
            self.tree.insert("", "end", text="Skewness of LBP", values=(skew_lbp,))  # Add this line
            self.tree.insert("", "end", text="Kurtosis of LBP", values=(kurtosis_lbp,))  # Add this line
            self.tree.insert("", "end", text="Chi-square statistic", values=(chi_square_stat,))  # Modify this line
            self.tree.insert("", "end", text="Sample Pair Analysis", values=(spa_stat,))  # Modify this line
        else:
            # Perform ML analysis
            svm = joblib.load('svm_model.joblib')  # Load the SVM
            # Calculate the RS Analysis features
            predicted_matrix = steganalysis.median_edge_detector(image_matrix)
            residuals = steganalysis.calculate_residuals(image_matrix, predicted_matrix)
            features_rs = steganalysis.calculate_rs_features(residuals)

            # Calculate the LBP features
            lbp = steganalysis.calculate_lbp(image_matrix)
            features_lbp = steganalysis.calculate_lbp_features(lbp)

            # Calculate the Chi-square attack feature
            features_chi = np.array([steganalysis.chi_square_attack(image_matrix)])  # Wrap the scalar in a 1D array

            # Calculate the Sample Pair Analysis feature
            features_spa = np.array([steganalysis.sample_pair_analysis(image_matrix)])  # Wrap the scalar in a 1D array

            # Concatenate the features into a single feature vector
            features = np.concatenate((features_rs, features_lbp, features_chi, features_spa))

            # Reshape the feature vector to a 2D array
            features = features.reshape(1, -1)  # Add this line

            prediction = svm.predict(features)  # Classify the image

            # Show the result in a messagebox
            messagebox.showinfo("SVM prediction", f"The SVM predicted the class {prediction[0]} for the image.")

if __name__ == "__main__":
    app = Application()
    app.mainloop()