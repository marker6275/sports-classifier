import mss
from PIL import Image, ImageTk
import tkinter as tk
import os

def read_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        return img

class ScreenViewer:
    def __init__(self, root, update_interval=1000, capture_interval=10, save_folder=""):
        """
        Args:
            root: the root window
            update_interval: the interval in milliseconds to update the image
            capture_interval: the interval in frames to capture the image
            save_folder: the folder to save the images

            Interval between captures: capture_interval * update_interval
            Example: capture_interval=10, update_interval=1000 -> 10000ms = 10s = capture every 10 seconds/every 10 frames
        """

        self.root = root
        self.root.title("Screen")

        self.update_interval = update_interval
        self.capture_interval = capture_interval
        self.save_folder = save_folder
        
        os.makedirs(self.save_folder, exist_ok=True)

        self.label = tk.Label(root)
        self.label.pack()

        self.tk_image = None

        self.picture_index = 0

        self.update_image()

    def update_image(self):
        img = read_screen() # reads the screen

        # take a picture of every 5 frames at 30 ms
        if self.picture_index % self.capture_interval == 0:
            filepath = os.path.join(self.save_folder, f"picture_{self.picture_index // self.capture_interval}.png")
            
            img.save(filepath)
        
        self.picture_index += 1

        self.tk_image = ImageTk.PhotoImage(img) # from PIL image to tkinter image

        self.label.config(image=self.tk_image) # displays the image

        self.root.after(self.update_interval, self.update_image) # after update_interval, update the image

if __name__ == "__main__":
    root = tk.Tk()

    app = ScreenViewer(root, update_interval=100, capture_interval=25, save_folder="screenshots") # Captures every 2.5 seconds/every 25 frames

    root.mainloop()