import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def extract_iteration(filename):
    """
    Extracts the iteration number from the filename for sorting.
    Example: 'training_400.png' -> 400
    """
    match = re.search(r'training_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return -1

def create_animation(solution_title='v(x,t)'):
    """
    Creates an animation from all available training images for a given solution.
    """
    # Define the base path to the folder containing the images
    base_folder_path = 'images'
    target_folder = os.path.join(base_folder_path, solution_title)
    
    print(f"Looking for images in: {target_folder}")

    # Make sure base folder exists before trying to read from it
    if not os.path.exists(target_folder):
        print(f"Error: The folder '{target_folder}' does not exist.")
        print("Please run training with the '--prep_anim' flag first.")
        return

    # Find all PNG files in the directory
    image_pattern = os.path.join(target_folder, 'training_*.png')
    image_paths = glob.glob(image_pattern)
    
    if not image_paths:
        print(f"No training images found in '{target_folder}'.")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")

    # Sort the files numerically by iteration number
    image_paths.sort(key=extract_iteration)

    fig, ax = plt.subplots(figsize=(14, 7))
    # Turn off axis to just show the image purely
    ax.axis('off')

    # Create a list to store the frames
    ims = []

    # Loop through the sorted image paths and add to the list of frames
    for path in image_paths:
        img = plt.imread(path)
        # using animated=True is necessary for ArtistAnimation
        im = ax.imshow(img, animated=True)
        ims.append([im])

    # Create an animation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

    # Define the output path for the GIF
    # Replace characters that might cause file system issues
    safe_title = solution_title.replace('(', '_').replace(')', '').replace(',', '_')
    output_path = os.path.join(base_folder_path, f'{safe_title}_animation.gif')

    # Save the animation as a GIF
    try:
        print(f"Saving animation. This might take a moment depending on the number of frames...")
        ani.save(output_path, writer='pillow')
        print(f'Success! GIF saved at: {output_path}')
    except Exception as e:
        print(f'Could not save GIF. Error: {e}')
    finally:
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF animation from training progress images.")
    parser.add_argument('--target', type=str, default='v(x,t)', 
                        help="The solution folder to animate (e.g., 'u(x,t)', 'v(x,t)', 'h(x,t)')")
    parser.add_argument('--all', action='store_true', 
                        help="Animate all standard outputs: u(x,t), v(x,t), and h(x,t)")
    
    args = parser.parse_args()
    
    if args.all:
        targets = ['u(x,t)', 'v(x,t)', 'h(x,t)']
        for target in targets:
            print("-" * 40)
            create_animation(target)
    else:
        create_animation(args.target)
