from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import Button
from tkinter import ttk
import os


def PopFileChoose():
    root = tk.Tk()
    root.withdraw()
    FilePath = filedialog.askopenfilename()
    return FilePath

def PopFileDir():
    root = tk.Tk()
    root.withdraw()
    FileDir = filedialog.askdirectory()
    return FileDir

def PopModelSelect():
    root = tk.Tk()
    root_x = root.winfo_rootx()
    root_y = root.winfo_rooty()
    win_x = root_x + 300
    win_y = root_y + 100
    root.geometry(f'+{win_x}+{win_y}')
    def my_upd(*args):
        l1.config(text=sel.get())
    def Close():
        root.destroy()
    sel=tk.StringVar() # string variable
    model=['Autoencoder']
    cb1 = ttk.Combobox(root, values=model, width=15, textvariable=sel)
    exitButton = Button(root, text='Select', command=Close)
    cb1.grid(row=1,column=1,padx=10,pady=10)
    exitButton.grid(row=1, column=3, padx=5, pady=10)
    l1=tk.Label(root, text='model')
    l1.grid(row=1,column=2)
    sel.trace('w',my_upd)
    root.mainloop()
    return sel.get()

def CheckPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def LoadModel(path):
    model = keras.models.load_model(path)
    return model