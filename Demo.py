from PIL import Image
import os

# Set path to your image dataset folder
path = 'C:\\Users\\zeesh\\OneDrive\\Documents\\Projects\\Mini_Project B-Tech\\XRayImages\\Knee X-ray Images\\MedicalExpert-I'

# Set the target size for your images (e.g., 256 x 256)
target_size = (256, 256)

# Loop over the subfolders in your dataset folder
for foldername in os.listdir(path):
    # Create a new folder to store preprocessed images
    new_foldername = 'preprocessed_' + foldername
    if not os.path.exists(new_foldername):
        os.mkdir(new_foldername)
    # Loop over the images in the current subfolder
    for filename in os.listdir(os.path.join(path, foldername)):
        # Open the image file
        img = Image.open(os.path.join(path, foldername, filename))
        # Resize the image to the target size
        img = img.resize(target_size)
        # Save the preprocessed image to the new folder
        img.save(os.path.join(new_foldername, filename))