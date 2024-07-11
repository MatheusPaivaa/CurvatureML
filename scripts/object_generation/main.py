import os
import subprocess
import include.obj_generator as go

# Set environment variable to suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_destination_folder(destination_folder):
    """
    Checks if the destination folder exists, and creates it if it doesn't.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

def check_images_folder(images_folder):
    """
    Checks if the images folder exists.
    """
    if not os.path.exists(images_folder):
        print(f"\nThe images folder '{images_folder}' does not exist. ")
        return False
    return True

def process_images(images_folder, destination_folder):
    """
    Processes images from the images folder and saves them to the destination folder.
    """
    # Check if the images folder exists before proceeding
    if not check_images_folder(images_folder):
        print("Please check the provided path.\n")
        return  # Interrupts execution if the images folder does not exist
    
    # Create the destination folder if necessary
    create_destination_folder(destination_folder)
    
    print(f"\nProcessing images from '{images_folder}' and saving to '{destination_folder}'\n")

    go.process_images(images_folder, destination_folder)

    print(f"\n\n.obj file generation completed! Processed images are in the folder '{destination_folder}'.\n")

    pass

if __name__ == "__main__":
    print("Welcome to the OBJ file generator.\n")

    images_folder = input("Enter the path to the images folder: ")
    process_images(images_folder, '../../data/face_processed')
