import os

def rename_photos(directory):
    """
    Renames photos in the specified directory to a sequential format.

    Parameters:
    directory (str): The path to the directory containing the photos.
    """
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Filter only the files that are images (common extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ppm']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    # Sort the images alphabetically
    images.sort()
    
    # Rename each image in ascending order
    for index, image in enumerate(images):
        new_name = f'face_{index+1:03d}{os.path.splitext(image)[1].lower()}'  # face_001.jpg, face_002.png, etc.
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')

# Specify the directory where your photos are located
directory = 'face_raw_1'
rename_photos(directory)
