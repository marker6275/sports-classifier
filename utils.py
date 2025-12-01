import os

def rename_files(folder, new_name, start_index=-1):
    # Create target directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    for file in os.listdir(folder):
        if file.endswith(".png"):
            file_index = start_index if start_index != -1 else file.split("_")[1].split(".")[0]
            if start_index != -1:
                start_index += 1
            
            new_filename = f"{new_name}-{file_index}.png"
            print(f"Renaming {file} - {file_index} to {new_filename}")

            if os.path.exists(os.path.join(folder, new_filename)):
                print(f"Error: File {new_filename} already exists in {folder}")
                continue

            os.rename(os.path.join(folder, file), os.path.join(folder, new_filename))

if __name__ == "__main__":
    rename_files("screenshots", "texans-chiefs", 0)