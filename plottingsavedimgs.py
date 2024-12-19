#%%
from PIL import Image
import os
#%%
directory = "/Users/kierapond/Documents/Documents/Graduate/Committee Meeting Report 2024/pythonimgs/2dpipe-changingheight-xdir"
# print(os.listdir(directory))

# List of PNG file paths
image_files = [directory+'/nobulk.png',
               directory+"/mup01-etap01.png", 
               directory+ "/mup01-eta1.png",
              directory+"/essentiallynoshear.png",]

# Open all images
images = [Image.open(img) for img in image_files]
#%%
### Putting pngs in a grid
# Determine the size of the grid
width, height = images[0].size
grid_width = 2 * width
grid_height = 2 * height

# Create a blank canvas for the grid
grid = Image.new("RGB", (grid_width, grid_height))

# Paste the images into the grid
grid.paste(images[0], (0, 0))                  # Top-left
grid.paste(images[1], (width, 0))             # Top-right
grid.paste(images[2], (0, height))            # Bottom-left
grid.paste(images[3], (width, height))        # Bottom-right

grid.show()
#%%
# Save the final grid
output_path = directory+"/increasingbulk_grid.png"
grid.save(output_path, format="PNG")
#%%
### Putting pngs in a line
# Get the total width and maximum height of the row
total_width = sum(img.width for img in images)
max_height = max(img.height for img in images)

# Create a blank canvas for the final image
combined_image = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))

# Paste each image side by side on the canvas
x_offset = 0
for img in images:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width

# Save or show the combined image
combined_image.show()
# %%
# Save the image as PNG
save_path = directory+"/increasingbulk_line.png"
combined_image.save(save_path, format='PNG')
# %%
