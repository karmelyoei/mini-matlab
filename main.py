# main.py
import tkinter as tk
from mini_matlab import MiniMatlab

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniMatlab(root)
    root.mainloop()
