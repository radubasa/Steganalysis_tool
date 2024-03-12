import tkinter as tk
from tkinter import messagebox

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("My Application")
        self.geometry("300x200")

        tk.Label(self, text="Hello to my application!").pack()

        tk.Button(self, text="Email Functionality", command=self.email_functionality).pack()
        tk.Button(self, text="Info", command=self.show_info).pack()
        tk.Button(self, text="Separate Functionalities", command=self.separate_functionalities).pack()

    def email_functionality(self):
        # Implement your email functionality here
        print("Email Functionality")

    def show_info(self):
        messagebox.showinfo("Info", "This is some information about the application.")

    def separate_functionalities(self):
        # Implement your separate functionalities here
        print("Separate Functionalities")

if __name__ == "__main__":
    app = Application()
    app.mainloop()