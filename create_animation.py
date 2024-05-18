import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(14, 7))

# Create a list to store the frames
ims = []

# Define the base path to the folder containing the images
base_folder_path = 'images'
solution_title = 'v(x,t)'  # Replace with your actual solution title

# Loop through the specified range to read images and add to the list of frames
for i in range(6000):
    if i % 40 == 0:
        path = os.path.join(base_folder_path, solution_title, f'training_{i}.png')
        if os.path.exists(path):
            img = plt.imread(path)
            im = ax.imshow(img, animated=True)
            ims.append([im])

# Create an animation
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

# Define the output path for the GIF
output_path = os.path.join(base_folder_path, 'v(x,t).gif')

# Save the animation as a GIF
ani.save(output_path, writer='Pillow')

print(f'GIF saved at {output_path}')
