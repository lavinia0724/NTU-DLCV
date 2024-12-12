from PIL import Image
import os

# Set folder path and image grid configuration
# folder_path = "./hw2/DDPM/output_images/mnistm"  # Folder containing images
folder_path = "./hw2/DDPM/output_images/svhn"
num_rows, num_cols = 10, 10
output_size = 280  # Output image dimensions (width and height)

# Create a blank output image
output_image = Image.new('RGB', (output_size, output_size))

# Loop through each position in the grid
for col in range(num_cols):
    for row in range(num_rows):
        image_filename = f"{col}_{row+1:03d}.png"  # Format of image filenames
        image_path = os.path.join(folder_path, image_filename)

        try:
            # Load the current image
            current_image = Image.open(image_path)
        except IOError:
            print(f"Cannot open image: {image_filename}")
            continue

        # Resize the image to fit into the grid cell
        cell_width = output_size // num_cols
        cell_height = output_size // num_rows
        current_image = current_image.resize((cell_width, cell_height), Image.LANCZOS)

        # Calculate the position for the current image in the output
        position_x = row * cell_width
        position_y = col * cell_height

        # Paste the resized image into the output image
        output_image.paste(current_image, (position_x, position_y))

# Save the final combined grid image
# output_image.save("P1_GeneratedGridImg_MNIST.png")
output_image.save("P1_GeneratedGridImg_SVHN.png")
