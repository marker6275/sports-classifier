#!/usr/bin/env python3
"""
Interactive image labeling tool for reviewing and labeling captured screenshots.
Displays images from data/unlabeled/ and allows you to label them.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import shutil
import json
import random
import os

DATA_DIR = Path("data")
UNLABELED_DIR = Path("screenshots")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
METADATA_FILE = DATA_DIR / "labeling_metadata.json"

# 65% train, 35% validation split
TRAIN_SPLIT = 0.65

CLASSES = ["basketball", "football", "commercial"]
IMAGE_SIZE = 800

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler - Sports Identifier")
        self.root.geometry("1000x900")
        
        self.current_index = 0
        self.image_files = []
        self.current_image = None
        self.current_label = None
        
        self.load_metadata()
        self.load_images()
        
        self.setup_ui()
        self.load_image(0)
        
    def load_metadata(self):
        """Load labeling metadata (skipped images, etc.)"""
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"skipped": []}
    
    def save_metadata(self):
        """Save labeling metadata"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_images(self):
        """Load all unlabeled images"""
        if not UNLABELED_DIR.exists():
            UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
        
        self.image_files = sorted(UNLABELED_DIR.glob("*.png"))
        self.image_files = [f for f in self.image_files if f.name not in self.metadata.get("skipped", [])]
        
        if not self.image_files:
            messagebox.showinfo("No Images", "No unlabeled images found in /screenshots")
            self.root.quit()
    
    def setup_ui(self):
        """Set up the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(
            info_frame, 
            text=f"Image 0 of {len(self.image_files)}",
            font=("Arial", 12, "bold")
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.prediction_label = ttk.Label(
            info_frame,
            text="",
            font=("Arial", 10)
        )
        self.prediction_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2)
        
        ttk.Button(
            button_frame,
            text="← Previous",
            command=self.prev_image,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        for i, class_name in enumerate(CLASSES):
            # Higher contrast colors
            if class_name in ["basketball", "football"]:
                bg_color = "#2E7D32"  # Darker green for better contrast
                fg_color = "#000000"  # Black text
            else:
                bg_color = "#C62828"  # Darker red for better contrast
                fg_color = "#000000"  # Black text
            
            btn = tk.Button(
                button_frame,
                text=class_name.upper(),
                command=lambda c=class_name: self.label_image(c),
                bg=bg_color,
                fg=fg_color,
                font=("Arial", 12, "bold"),
                width=15,
                height=2,
                relief=tk.RAISED,
                borderwidth=3,
                activebackground=bg_color,
                activeforeground=fg_color
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Skip",
            command=self.skip_image,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Next →",
            command=self.next_image,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        keyboard_frame = ttk.Frame(main_frame)
        keyboard_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Label(
            keyboard_frame,
            text="Keyboard shortcuts: 1=Basketball, 2=Football, 3=Commercial, ←=Previous, →=Next, Space=Skip",
            font=("Arial", 9),
            foreground="gray"
        ).pack()
        
        self.root.bind("<Key-1>", lambda e: self.label_image("basketball"))
        self.root.bind("<Key-2>", lambda e: self.label_image("football"))
        self.root.bind("<Key-3>", lambda e: self.label_image("commercial"))
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.skip_image())
        self.root.focus_set()
    
    def load_image(self, index):
        """Load and display an image"""
        if not (0 <= index < len(self.image_files)):
            return
        
        self.current_index = index
        image_path = self.image_files[index]
        
        img = Image.open(image_path)
        img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        self.status_label.configure(
            text=f"Image {index + 1} of {len(self.image_files)} - {image_path.name}"
        )
        
        self.current_image = image_path
        self.current_label = None
    
    def label_image(self, label):
        """Label the current image and move it to train or val directory with 65/35 split"""
        if not self.current_image:
            messagebox.showerror("Error", "No current image selected")
            return
        
        # Check if source file exists
        source_path = self.current_image.resolve()  # Get absolute path
        if not source_path.exists():
            error_msg = f"Source file not found:\n{source_path}"
            messagebox.showerror("Error", error_msg)
            return
        
        try:
            # Randomly assign to train (65%) or val (35%)
            split = "train" if random.random() < TRAIN_SPLIT else "val"
            split_dir = TRAIN_DIR.resolve() if split == "train" else VAL_DIR.resolve()
            
            # Create directories
            split_dir.mkdir(parents=True, exist_ok=True)
            label_dir = split_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = label_dir / self.current_image.name
            
            if dest_path.exists():
                if not messagebox.askyesno("Overwrite?", f"File already exists in {split}/{label}. Overwrite?"):
                    return
            
            # Move the file - use absolute paths to be safe
            source_str = str(source_path.resolve())
            dest_abs = dest_path.resolve()
            dest_str = str(dest_abs)
            
            # Verify source exists before moving
            if not Path(source_str).exists():
                raise Exception(f"Cannot move: source file does not exist: {source_str}")
            
            # Perform the move
            shutil.move(source_str, dest_str)
            
            # Verify the move succeeded
            dest_exists = Path(dest_str).exists()
            source_still_exists = Path(source_str).exists()
            
            if source_still_exists and not dest_exists:
                # Move failed - try copy + delete instead
                try:
                    shutil.copy2(source_str, dest_str)
                    if Path(dest_str).exists():
                        os.remove(source_str)
                        dest_exists = True
                    else:
                        raise Exception("Copy failed - destination still doesn't exist")
                except Exception as copy_error:
                    raise Exception(f"Both move and copy failed: {copy_error}")
            
            if not dest_exists:
                actual_files = list(dest_path.parent.glob("*"))
                raise Exception(
                    f"File move verification failed!\n"
                    f"Expected: {dest_abs}\n"
                    f"Files in {dest_path.parent}: {[f.name for f in actual_files]}"
                )
            
            if source_still_exists:
                try:
                    os.remove(source_str)
                except Exception:
                    pass  # File was moved successfully even if source remains
            
            # Verify the file is at the destination
            moved_file = dest_path if dest_path.exists() else dest_abs
            if not moved_file.exists():
                # Search for the file
                found_files = list(dest_path.parent.glob(self.current_image.name))
                if found_files:
                    moved_file = found_files[0]
                else:
                    # Search entire data directory tree
                    all_matches = list(DATA_DIR.resolve().glob(f"**/{self.current_image.name}"))
                    if all_matches:
                        moved_file = all_matches[0]
                    else:
                        actual_files = list(dest_path.parent.glob("*"))
                        raise Exception(
                            f"File not found after move!\n"
                            f"Expected: {dest_abs}\n"
                            f"Files in {dest_path.parent}: {[f.name for f in actual_files]}"
                        )
            
            # Print move confirmation
            print(f"Moved {self.current_image.name} from screenshots to {split}/{label}/")
            
            self.root.update()  # Force UI update
            
            # Remove from image list BEFORE loading next image
            self.image_files.pop(self.current_index)
            
            # Adjust index if needed
            if self.current_index >= len(self.image_files):
                self.current_index = max(0, len(self.image_files) - 1)
            
            # Small delay to show the status message
            self.root.after(500, lambda: self._load_next_after_label())
                
        except Exception as e:
            error_msg = f"Failed to move image:\n{str(e)}\n\nSource: {source_path}"
            messagebox.showerror("Error", error_msg)
    
    def _load_next_after_label(self):
        """Load next image after labeling"""
        if self.image_files:
            self.load_image(self.current_index)
        else:
            messagebox.showinfo("Done", "All images have been labeled!")
            self.root.quit()
    
    def skip_image(self):
        """Skip the current image (mark as skipped)"""
        if not self.current_image:
            return
        
        if self.current_image.name not in self.metadata["skipped"]:
            self.metadata["skipped"].append(self.current_image.name)
            self.save_metadata()
        
        self.next_image()
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.load_image(self.current_index + 1)
        else:
            messagebox.showinfo("End", "Reached the end of the image list.")

if __name__ == "__main__":
    root = tk.Tk()
    ImageLabeler(root)
    root.mainloop()

