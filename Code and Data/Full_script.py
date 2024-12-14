import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt

# Q4 shiny_vase dataset - The code up until RANSAC stems from the beethoven_run-.py file and has been modified to fit Q4 for the assignment.
#Load data
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


#Display the results from applying Lamberts law - which the dataset does not abide by as it introduces specularities 
_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.show()

z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)

ps_utils.display_surface(z)


#Now try with RANSAC
#Initialise rays for albedo and normale using RANSAC
M_ransac = np.zeros_like(M)

#Loop through each pixel to apply RANSAC (adapted for each pixel's data)
for i in range(len(nz[0])):
    pixel_intensities = J[:,i]
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, 2.0)
    M_ransac[:, i] = normal * np.linalg.norm(pixel_intensities)

rho_ransac = np.linalg.norm(M_ransac, axis=0) + 1e-8  # Prevent zero division
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
ax1.imshow(n1_ransac, cmap='coolwarm')
ax1.set_title("n1 (RANSAC)")
ax2.imshow(n2_ransac, cmap='coolwarm')
ax2.set_title("n2 (RANSAC)")
ax3.imshow(n3_ransac, cmap='coolwarm')
ax3.set_title("n3 (RANSAC)")
plt.show()


# Smooth the normal field using `smooth_normal_field`
n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
    n1_ransac, n2_ransac, n3_ransac, mask, iters=15
)

# Display the smoothed results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_smooth, cmap='coolwarm')
ax1.set_title("n1 (Smoothed)")
ax2.imshow(n2_smooth, cmap='coolwarm')
ax2.set_title("n2 (Smoothed)")
ax3.imshow(n3_smooth, cmap='coolwarm')
ax3.set_title("n3 (Smoothed)")
plt.show()



#Q6
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


#Display the results from applying Lamberts law - which the dataset does not abide by as it introduces specularities 
_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.show()

z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
z = np.nan_to_num(z)

ps_utils.display_surface(z)


#Now try with RANSAC
#Initialise rays for albedo and normale using RANSAC
M_ransac = np.zeros_like(M)

#Loop through each pixel to apply RANSAC (adapted for each pixel's data)
for i in range(len(nz[0])):
    pixel_intensities = J[:,i]
    IS = (pixel_intensities, S)
    normal, _, _ = ps_utils.ransac_3dvector(IS, 25.0)
    M_ransac[:, i] = normal * np.linalg.norm(pixel_intensities)

rho_ransac = np.linalg.norm(M_ransac, axis=0) + 1e-8  # Prevent zero division
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
ax1.imshow(n1_ransac, cmap='coolwarm')
ax1.set_title("n1 (RANSAC)")
ax2.imshow(n2_ransac, cmap='coolwarm')
ax2.set_title("n2 (RANSAC)")
ax3.imshow(n3_ransac, cmap='coolwarm')
ax3.set_title("n3 (RANSAC)")
plt.show()


# Smooth the normal field using `smooth_normal_field`
n1_smooth, n2_smooth, n3_smooth = ps_utils.smooth_normal_field(
    n1_ransac, n2_ransac, n3_ransac, mask, iters=15
)

# Display the smoothed results
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(n1_smooth, cmap='coolwarm')
ax1.set_title("n1 (Smoothed)")
ax2.imshow(n2_smooth, cmap='coolwarm')
ax2.set_title("n2 (Smoothed)")
ax3.imshow(n3_smooth, cmap='coolwarm')
ax3.set_title("n3 (Smoothed)")
plt.show()
