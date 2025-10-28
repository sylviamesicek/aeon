import pyvista as pv

def axi_evolve_anim(dir, name):
    """
    Produces an animation for a given evolution
    """
    plotter = pv.Plotter(notebook=False, off_screen=True)
    mesh = pv.read(f"{dir}/{name}_0.vtu")
    plotter.add_mesh(mesh, cmap='viridis', scalars='alpha')
    plotter.camera_position = 'iso' # Set camera position if desired
    plotter.render()
    plotter.screenshot(f"frame_{0:04d}.png") # Save each frame as a PNG image
    plotter.clear_meshes() # Clear the mesh for the next frame