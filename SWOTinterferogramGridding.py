# Import libraries
import glob
import numpy as np
import netCDF4 as nc
import scipy

# Load files
files_list = glob.glob("D:\MasterThesis\PreBetaKaRIn_HighRate\*_016_220L*.nc")
print("Total amount of files:",len(files_list))

for i_file, filename in enumerate(files_list):
    print("{}/{}".format(i_file, len(files_list)))
    print("File now processing:", filename)

    # Check if file exists
    if glob.glob(filename[:-3]+"_grid.npy"):
        print("File exists, skipping")
        continue

    #### Load data
    high_rate = nc.Dataset(filename, "r")
    
    lon = high_rate["pixel_cloud"]["longitude"][:]
    lat = high_rate["pixel_cloud"]["latitude"][:]
    h = high_rate["pixel_cloud"]["height"][:]
    geoid = high_rate["pixel_cloud"]["geoid"][:]


    #### Grid data

    # radar dimension
    p1 = np.array([high_rate.inner_first_longitude, high_rate.inner_first_latitude])
    p2 = np.array([high_rate.outer_first_longitude, high_rate.outer_first_latitude])
    p3 = np.array([high_rate.inner_last_longitude, high_rate.inner_last_latitude])
    p4 = np.array([high_rate.outer_last_longitude, high_rate.outer_last_latitude])

    # Parametrize the array
    v_range = p2 - p1
    v_along = p3 - p1

    # Determine stepsize
    grid_pixelsize = 25 # m
    along_diff = (p4 - p2) * np.array([40075*np.cos(np.deg2rad(p2[1])) / 360, 111.139])
    #print(along_diff)
    along_stepsize = int( (np.sqrt(np.sum(along_diff**2))*1000) / grid_pixelsize )
    #print(along_stepsize)

    range_diff = (p1 - p2) * np.array([40075*np.cos(np.deg2rad(p2[1])) / 360, 111.139])
    #print(range_diff)
    range_stepsize = int( (np.sqrt(np.sum(range_diff**2))*1000) / grid_pixelsize)
    #print(range_stepsize)
        
    # Create grid
    s_range = np.linspace(0,1, range_stepsize)
    s_along = np.linspace(0,1, along_stepsize)
    grid_shape = (s_along.size, s_range.size)
    #print("grid_size", grid_shape)

    p = p1 + [[s_ * v_range for s_ in s_range] + t_ * v_along for t_ in s_along]
    #print("points_size", p.shape)

    p = p.reshape(-1,2)

    #### Interpolate data
    ph = scipy.interpolate.griddata(np.c_[lon, lat], h, np.c_[p[:,0], p[:,1]])
    geoid_grid = scipy.interpolate.griddata(np.c_[lon, lat], geoid, np.c_[p[:,0], p[:,1]])
    
    # Reshape data onto grid
    ph_grid = ph.reshape(grid_shape)
    lat_grid = p[:,1].reshape(grid_shape)
    lon_grid = p[:,0].reshape(grid_shape)
    geoid_grid = geoid_grid.reshape(grid_shape)

    # Combine data
    data = np.dstack([lat_grid, lon_grid, ph_grid, geoid_grid])

    # Save data
    np.save(filename[:-3]+"_grid", data)
    print("File saved as ",filename[:-3]+"_grid")