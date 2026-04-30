import h5py
import os


FILE_PATH = r'data\raw\Train\dataset_training_no_aug.h5'

def explore_h5_file(file_path):
    print(f"--- Exploring: {os.path.basename(file_path)} ---")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please check your folder structure!")
        return

   
    with h5py.File(file_path, 'r') as f:
        print("Keys inside the H5 file:", list(f.keys()))
        
       
        for key in f.keys():
            data = f[key]
            print(f"\n--- Key: '{key}' ---")
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            
          
            print(f"First item preview:\n{data[0]}")

if __name__ == "__main__":
    explore_h5_file(FILE_PATH)