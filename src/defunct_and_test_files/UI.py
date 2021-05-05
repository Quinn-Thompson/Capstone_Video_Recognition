import tkinter as tk
from DBM import databaseAccess
from PIL import Image, ImageTk
import numpy as np
from middleware import Middelware
import time
from ModelWrapper import CapModel

cm = CapModel()

# create the root window
root = tk.Tk()

# the database obj
dbm = databaseAccess()

# the middleware object
mw = Middelware()

# root window title and dimension
root.title("OSU CS Capstone - ML Gest Rec")
# Set geometry(widthxheight)
root.geometry('840x480')

# Add image file
bg = ImageTk.PhotoImage(file="bk.jpg")

# Create Canvas
bkg = tk.Canvas(root, width=840, height=480)

bkg.pack(fill="both", expand=True)

# Display image
bkg.create_image(0, 0, image=bg, anchor="nw")


depth = tk.Canvas(root, width=320, height=240)
depth.place(x=40, y=20)
rgb = tk.Canvas(root, width=320, height=240)
rgb.place(x=840-40-320, y=20)

depthLabel = tk.Label(root, text="Raw Depth Image")
depthLabel.place(x=200, y=280, anchor="center")
colorLabel = tk.Label(root, text="Proccessed Depth Image")
colorLabel.place(x=840-40-160, y=280, anchor="center")

#predictiveTxtLbl = tk.Label(root, text="Predictive Text: ")
##predictiveTxtLbl.place(x=200, y=320, anchor="e")

predictiveChr1Lbl = tk.Label(root, text="First Predictive Character: ")
predictiveChr1Lbl.place(x=200, y=340, anchor="e")

confChr1Lbl = tk.Label(root, text="Confidence: ")
confChr1Lbl.place(x=200, y=360, anchor="e")

predictiveChr2Lbl = tk.Label(root, text=" ")
predictiveChr2Lbl.place(x=200, y=340, anchor="w")

confChr2Lbl = tk.Label(root, text=" ")
confChr2Lbl.place(x=200, y=360, anchor="w")

flowCntrlBtn = tk.Button(text="Start/Stop")
flowCntrlBtn.place(x=840-40-160, y=340, anchor="center")

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Classification")
filemenu.add_command(label="Data Collection")
filemenu.add_command(label="About")






menubar.add_cascade(label="Options", menu=filemenu)
root.config(menu=menubar)

# Execute Tkinter
#root.mainloop()
while(1):
    img = ImageTk.PhotoImage(image=Image.fromarray(
                             mw.returnDeptInt(size=(240, 320))*255))
    imgPP = ImageTk.PhotoImage(image=Image.fromarray(
                             mw.returnDeptIntPP(size=(240, 320))*255))
    imgClss = mw.returnDeptIntPP(size=(48, 64))

    lbl, conf = cm.Classify(np.array(imgClss))

    predictiveChr2Lbl.config(text=str(lbl))
    confChr2Lbl.config(text=str(conf))

    depth.create_image(0, 0, anchor='nw', image=img)
    rgb.create_image(0, 0, anchor='nw', image=imgPP)
    mw.update()
    root.update_idletasks()
    root.update()
    time.sleep(0.01)
