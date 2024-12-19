#%%
from PIL import Image
import os
#%%
directory = "/Users/kierapond/Documents/Documents/Graduate/Committee Meeting Report 2024/pythonimgs/2dpipe-changingheight-xdir"
# print(os.listdir(directory))

# List of PNG file paths
image_files = [directory+"/essentiallynoshear.png", 
               directory+"/mup01-etap01.png", 
               directory+'/nobulk.png',
              directory+ "/mup01-eta1.png"]

# Open all images
images = [Image.open(img) for img in image_files]

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
combined_image.save("row_of_images.png")
combined_image.show()

# %%
