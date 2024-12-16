## Assignment 3: Photometric Stereo
## Francesca Salute --> bhn327
## Martin Havresøe --> lwz885
## Nicole Favero --> sxr554 

# import libraries
import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed Working Directory:", os.getcwd())

# The code up until RANSAC stems from the beethoven_run-.py file and has been modified to adapt to the requirements in each task of the assignment.

# --------------------------------------------------------------------------
# Task 2: Beethoven Dataset 
# --------------------------------------------------------------------------

# Load data for Beethoven dataset
I, mask, S = ps_utils.read_data_file('Beethoven.mat')

# Extract indices of non-zero pixels in the mask
nz = np.where(mask > 0)
m, n = mask.shape

# Collect intensity data (J) only for valid pixels:
# J is a 3 x nz matrix, where each column corresponds to the intensity values for a pixel across the 3 images.
J = np.zeros((3, len(nz[0])))
for i in range(3):
    Ii = I[:, :, i]
    J[i, :] = Ii[nz]

# Solve for M (Albedo-Modulated Normal Field) using Woodham's idea
# Using Lambert's law: I = rho * (S @ n), solve for M = S^-1 * J
iS = la.inv(S)  # Inverse of S since S is square (3x3) for this dataset 
M = np.dot(iS, J)  # M = S^-1 * J

# Compute Albedo (rho) and Normalize M to get Surface Normals (N)
rho = la.norm(M, axis=0)
N = M / np.tile(rho, (3, 1))

# Map the normal components back into the image space
n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.zeros((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]

# Display Albedo as a 2D image
albedo_image = np.zeros((m, n))
albedo_image[nz] = rho
plt.imshow(albedo_image, cmap='gray')
plt.title("Albedo")
plt.colorbar()
plt.show()

# Display normal field components
_, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.suptitle("Surface normal components for Beethoven Dataset") 
plt.show()

# Compute Depth Map using Normal Field
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)  # Solve Poisson equation
z = np.nan_to_num(z)  # Handle NaN values

# Display surface at different view points
# Default viewpoint
ps_utils.display_surface(z)

# Define x and y for 3D plotting
m, n = z.shape  # Dimensions of the depth map
x, y = np.meshgrid(np.arange(n), np.arange(m))

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=-50, azim=45)  
plt.show()

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=70, azim=90)  
plt.show()

# --------------------------------------------------------------------------
# Task 3: mat_vase Dataset
# --------------------------------------------------------------------------

# loaded using function provided in ps_utils
I, mask, S = ps_utils.read_data_file('mat_vase.mat')  

nz = np.where(mask > 0) # pixels in non-zero part
m, n = mask.shape
J = np.zeros((3, len(nz[0]))) # array J for the non-zero region
for i in range(3):
    Ii = I[:, :, i]  # extract images
    J[i, :] = Ii[nz]  # assign the pixels substituting the zeros

# compute albedo modulated normal field 
Si = la.inv(S)  
M = np.dot(Si, J)
Rho = la.norm(M, axis=0)  # euclidean norm for albedo values

# extract the normal components
N = M / np.tile(Rho, (3, 1))  
# reshaping them to get an image
n1, n2, n3 = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
n1[nz] = N[0, :]  
n2[nz] = N[1, :]  
n3[nz] = N[2, :]  

# display image
albedo_image = np.zeros((m, n))
albedo_image[nz] = Rho  # use albedo values
plt.figure()
plt.title("Albedo Image")
plt.imshow(albedo_image, cmap='gray')
plt.colorbar()
plt.show()

# display normal components
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
components = [(n1, 'n1 (X-component)'), (n2, 'n2 (Y-component)'), 
              (n3, 'n3 (Z-component)')]
for ax, (data, title) in zip(axes, components):
    ax.set_title(title)
    ax.imshow(data)
plt.tight_layout()  
plt.show()

# solve for depth and display at multiple viewpoints
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)  # in case of nan values
ps_utils.display_surface(z)

# --------------------------------------------------------------------------
# Task 4: shiny vase Dataset
# --------------------------------------------------------------------------
# 
# #Load data
I, mask, S = ps_utils.read_data_file('shiny_vase.mat')

#Extract indices from the non-zero pixels in masks
nz = np.where(mask >0)
m,n = mask.shape

#For each mask pixel collect the image data
J = np.zeros((3,len(nz[0])))
for i in range(3):
    Ii = I[:,:,i]
    J[:i] = Ii[nz]

#Solve for M = rho*N
iS = la.inv(S)
M = np.dot(iS, J)

#Get albedo as a norm of M and normalise M
rho = la.norm(M, axis=0)
N = M/np.tile(rho, (3,1))

n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.zeros((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]


#Display the results  
_,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15, 5))
ax1.imshow(n1)
ax1.set_title("n1")
ax2.imshow(n2)
ax2.set_title("n2")
ax3.imshow(n3)
ax3.set_title("n3")
plt.show()

#compute depth map
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)

#default view
ps_utils.display_surface(z)

# Define x and y for 3D plotting
m, n = z.shape  # Dimensions of the depth map
x, y = np.meshgrid(np.arange(n), np.arange(m))

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=-50, azim=45)  
ax.set_title('View 2')
plt.show()

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=70, azim=90)  
ax.set_title('View 3')
plt.show()

#Now try with RANSAC
#Initialise rays for albedo and normale using RANSAC
M_ransac = np.zeros_like(M)

#Loop through each pixel to apply RANSAC (adapted for each pixel's data)
for i in range(len(nz[0])):
    pixel_intensities = J[:,i]
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, 2.0)
    M_ransac[:, i] = normal * la.norm(pixel_intensities)

rho_ransac = la.norm(M_ransac, axis=0) + 1e-8  # Prevent zero division
N_ransac = M_ransac / np.tile(rho_ransac, (3, 1))

# Fill the normal components into 2D arrays for visualization
n1_ransac = np.zeros((m, n))
n2_ransac = np.zeros((m, n))
n3_ransac = np.zeros((m, n))
n1_ransac[nz] = N_ransac[0, :]
n2_ransac[nz] = N_ransac[1, :]
n3_ransac[nz] = N_ransac[2, :]

# Display the RANSAC results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_ransac)
ax1.set_title("n1 (RANSAC)")
ax2.imshow(n2_ransac)
ax2.set_title("n2 (RANSAC)")
ax3.imshow(n3_ransac)
ax3.set_title("n3 (RANSAC)")
plt.show()


# Smooth the normal field using `smooth_normal_field`
n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
    n1_ransac, n2_ransac, n3_ransac, mask, iters=10
)

# Display the smoothed results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_smooth)
ax1.set_title("n1 (Smoothed RANSAC)")
ax2.imshow(n2_smooth)
ax2.set_title("n2 (Smoothed RANSAC)")
ax3.imshow(n3_smooth)
ax3.set_title("n3 (Smoothed RANSAC)")
plt.show()

# --------------------------------------------------------------------------
# Task 5: shiny vase Dataset2
# --------------------------------------------------------------------------
# 
# load data and prepare it
I, mask, S = ps_utils.read_data_file('shiny_vase2.mat') 

nz = np.where(mask > 0)  
m, n = mask.shape    
J = np.zeros((22, len(nz[0])))  
for i in range(22):
    Ii = I[:, :, i]  
    J[i, :] = Ii[nz]

# albedo with the pseudo inverse
iS = np.linalg.pinv(S)
M = np.dot(iS, J)
Rho = la.norm(M, axis=0)

# normal components and reshape
N = M / np.tile(Rho, (3, 1))
n1, n2, n3 = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
n1[nz] = N[0, :]
n2[nz] = N[1, :]
n3[nz] = N[2, :]

# display albedo
albedo_image = np.zeros((m, n))
albedo_image[nz] = Rho
plt.figure()
plt.title("Albedo Image")
plt.imshow(albedo_image, cmap='gray')
plt.colorbar()
plt.show()

# display normal components
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
components = [(n1, 'n1 (X-component)'), (n2, 'n2 (Y-component)'), 
              (n3, 'n3 (Z-component)')]
for ax, (data, title) in zip(axes, components):
    ax.set_title(title)
    ax.imshow(data, cmap='gray')
plt.tight_layout()
plt.show()

# depth and viewpoints
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)

ps_utils.display_surface(z)

# try RANSAC
# pixel by pixel operation

M_ransac = np.zeros_like(M)

for i in range(len(nz[0])):
    pixel_intensities = J[:,i]
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, 2.0)
    M_ransac[:, i] = normal * np.linalg.norm(pixel_intensities)

Rho_ransac = np.linalg.norm(M_ransac, axis=0) + 1e-8  
print(f"Rho_ransac[{i}] = {Rho_ransac[i]}, type: {type(Rho_ransac[i])}, shape: {np.shape(Rho_ransac[i])}")
N_ransac = M_ransac / np.tile(Rho_ransac, (3, 1))
print(f"N_ransac shape: {N_ransac.shape}")

# Fill the normal components into 2D arrays for visualization
n1_ransac = np.zeros((m, n))
n2_ransac = np.zeros((m, n))
n3_ransac = np.zeros((m, n))
n1_ransac[nz] = N_ransac[0, :]
n2_ransac[nz] = N_ransac[1, :]
n3_ransac[nz] = N_ransac[2, :]

# diplay ransac
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
components = [(n1_ransac, 'n1 (X-component, RANSAC)'), 
              (n2_ransac, 'n2 (Y-component, RANSAC)'), 
              (n3_ransac, 'n3 (Z-component, RANSAC)')]

for ax, (data, title) in zip(axes, components):
    ax.set_title(title)
    ax.imshow(data, cmap='gray')
plt.tight_layout()
plt.show()

# smooth field with different iters
for iters in [5, 15, 30]:
    n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
        n1_ransac, n2_ransac, n3_ransac, mask, iters=iters)
    
# diplay results
print(f"Results after smoothing with {iters} iterations:")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
components = [(n1_smooth, 'n1 (X-component, Smoothed)'), 
              (n2_smooth, 'n2 (Y-component, Smoothed)'), 
              (n3_smooth, 'n3 (Z-component, Smoothed)')]

for ax, (data, title) in zip(axes, components):
    ax.set_title(title)
    ax.imshow(data)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# Task 6: Buddha Dataset
# --------------------------------------------------------------------------

I, mask, S = ps_utils.read_data_file('Buddha.mat')

#Extract indices from the non-zero pixels in masks
nz = np.where(mask >0)
m,n = mask.shape

#For each mask pixel collect the image data
J = np.zeros((10,len(nz[0])))
for i in range(3):
    Ii = I[:,:,i]
    J[:i] = Ii[nz]

#Solve for M = rho*N
iS = np.linalg.pinv(S)
M = np.dot(iS, J)

#Get albedo as a norm of M and normalise M
rho = la.norm(M, axis=0)
N = M/np.tile(rho, (3,1))

n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.zeros((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]


#Display the results
_,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15, 5))
ax1.imshow(n1)
ax1.set_title("n1")
ax2.imshow(n2)
ax2.set_title("n2")
ax3.imshow(n3)
ax3.set_title("n3")
plt.show()

#compute depth map
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)

#default view
ps_utils.display_surface(z)

# Define x and y for 3D plotting
m, n = z.shape  # Dimensions of the depth map
x, y = np.meshgrid(np.arange(n), np.arange(m))

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=-50, azim=45)
ax.set_title('View 2')
plt.show()

# Display the surface from a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.view_init(elev=70, azim=90)
ax.set_title('View 3')
plt.show()

#Now try with RANSAC
#Initialise rays for albedo and normale using RANSAC
M_ransac = np.zeros_like(M)

#Loop through each pixel to apply RANSAC (adapted for each pixel's data)
for i in range(len(nz[0])):
    pixel_intensities = J[:,i]
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, 35.0)
    M_ransac[:, i] = normal * la.norm(pixel_intensities)

rho_ransac = la.norm(M_ransac, axis=0) + 1e-8  # Prevent zero division
N_ransac = M_ransac / np.tile(rho_ransac, (3, 1))

# Fill the normal components into 2D arrays for visualization
n1_ransac = np.zeros((m, n))
n2_ransac = np.zeros((m, n))
n3_ransac = np.zeros((m, n))
n1_ransac[nz] = N_ransac[0, :]
n2_ransac[nz] = N_ransac[1, :]
n3_ransac[nz] = N_ransac[2, :]

# Display the RANSAC results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_ransac)
ax1.set_title("n1 (RANSAC)")
ax2.imshow(n2_ransac)
ax2.set_title("n2 (RANSAC)")
ax3.imshow(n3_ransac)
ax3.set_title("n3 (RANSAC)")
plt.show()


# Smooth the normal field using `smooth_normal_field`
n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
    n1_ransac, n2_ransac, n3_ransac, mask, iters=10
)

# Display the smoothed results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_smooth)
ax1.set_title("n1 (Smoothed RANSAC)")
ax2.imshow(n2_smooth)
ax2.set_title("n2 (Smoothed RANSAC)")
ax3.imshow(n3_smooth)
ax3.set_title("n3 (Smoothed RANSAC)")
plt.show()


# --------------------------------------------------------------------------
# Task 7: Face Dataset
# --------------------------------------------------------------------------

# Load the Face Dataset
I, mask, S = ps_utils.read_data_file('face.mat')

# Extract indices of non-zero pixels in the mask
nz = np.where(mask > 0) 
m, n = mask.shape  

# Collect intensity data (J)
# J has shape (27, nz)
J = np.zeros((27, len(nz[0])))
for i in range(27):  
    Ii = I[:, :, i]
    J[i, :] = Ii[nz]

# RANSAC
#Initialise rays for albedo and normale using RANSAC
M_ransac = np.zeros((3, len(nz[0])))  
#Loop through each pixel to apply RANSAC (adapted for each pixel's data)
for i in range(len(nz[0])):  # Iterate over all non-zero pixels
    pixel_intensities = J[:, i]  
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, threshold=10.0)  # threshold = 10.0
    M_ransac[:, i] = normal * np.linalg.norm(pixel_intensities)  # Albedo-modulated normal

# Normalize M
rho_ransac = np.linalg.norm(M_ransac, axis=0) + 1e-8  # Prevent division by zero
N_ransac = M_ransac / np.tile(rho_ransac, (3, 1))

# Map normals back to the image space
n1_ransac = np.zeros((m, n))
n2_ransac = np.zeros((m, n))
n3_ransac = np.zeros((m, n))
n1_ransac[nz] = N_ransac[0, :]
n2_ransac[nz] = N_ransac[1, :]
n3_ransac[nz] = N_ransac[2, :]

# Display Normal Components
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_ransac)
ax2.imshow(n2_ransac)
ax3.imshow(n3_ransac)
plt.show()

#compute depth map
z = ps_utils.unbiased_integrate(n1_ransac, n2_ransac, n3_ransac, mask)
z = np.nan_to_num(z)

#default view
ps_utils.display_surface(z)

# Smooth the normal field with different 'iters' values
for iters in [5, 15, 30]:
    n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
        n1_ransac, n2_ransac, n3_ransac, mask, iters=iters
    )
    
    # Display the smoothed results
    print(f"Results after smoothing with {iters} iterations:")
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(n1_smooth)
    ax2.imshow(n2_smooth)
    ax3.imshow(n3_smooth)
    plt.show()

    z = ps_utils.unbiased_integrate(n1_smooth, n2_smooth, n3_smooth, mask)
    z = np.nan_to_num(z)

    #default view
    ps_utils.display_surface(z)

