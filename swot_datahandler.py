# Import libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy
import cartopy.crs as ccrs
import datetime


# ========================================================================
# Load Simulated KaRIn Data
def loadKarinSimulated(path, maxfiles=None):
    """
    Inputs:
        Path: str
        maxfiles:
    
    Outputs:
        simkarin: 
    """

    # Ex: "D:\MasterThesis\SimulatedKaRIn\*"
    EXTERNAL_DATA_PATH = path

    # Load filenames in path
    cycles_path = glob.glob(path)

    # Determine if we have a max filecount
    # If we have then reduce
    if maxfiles != None:
        cycles_path = cycles_path[:maxfiles]
    
    # Setup cycles for KaRIn dictionary from filenames in path
    simkarin = {}
    for i in range(len(cycles_path)):
        c = int(glob.glob(EXTERNAL_DATA_PATH)[i][52:56].replace("_",""))
        simkarin[c] = {}

    # For each cycle, load each pass
    for c in range(len(cycles_path)):
        # Print the current cycles we are loading
        print("Cycle {}/{}".format(c+1, len(cycles_path)),end="\r")

        # Get the filenames we are using in this cycle
        SWOT_filenames = [cycles_path[c]]

        # Setup the passes in the dictionary from the filenames
        karin_pass = {}
        for i in range(len(SWOT_filenames)):
            p = int(SWOT_filenames[i][57:60].replace("_",""))
            karin_pass[p] = karin_pass

        # For each pass, load the file into the dict
        for i, SWOT_GDR_filename in enumerate(SWOT_filenames):
            # Print the current file
            #print("File {}/{}".format(i+1, len(SWOT_filenames)),end="\r")
            file = nc.Dataset(SWOT_GDR_filename, "r")

            # Load data for each file
            keys = file.variables.keys()
            cyc = file.cycle_number
            pas = file.pass_number

            # Get data and load into the dict
            data_ = {}
            for k in keys:
                data_[k] = np.ma.getdata(file[k][:])
                karin_pass[pas] = data_
                simkarin[cyc] = karin_pass

            # Convert from 0,360 to -180,180 degrees longitude
            simkarin[cyc][pas]["longitude"][simkarin[cyc][pas]["longitude"]>180] -= 360

    # Pass the data
    return simkarin
# ========================================================================


# ========================================================================
# Load Poseidon-3C 20Hz Nadir Altimeter data
def loadNadir20Hz(path, maxfiles=None, cycles=[], passes=[]):
    """
    Inputs:
        Path: str
        maxfiles:
    
    Outputs:
        nadir: 
    """

    # Ex: "D:\MasterThesis\SIGDR\*"
    EXTERNAL_DATA_PATH = path

    # Load filenames in path
    cycles_path = glob.glob(path)

    # Reduce cycles path to only contain certain cycles
    if cycles != []:
        cycles_path = [s for s in cycles_path if "560" in s]

    # Determine if we have a max filecount
    # If we have then reduce
    if maxfiles != None:
        cycles_path = cycles_path[:maxfiles]

    # Setup cycles for Nadir dictionary from filenames in path
    nadir = {}
    for i in range(len(cycles_path)):
        c = int(cycles_path[i][-3:])
        nadir[c] = {}

    # For each cycle, load each pass
    for c in range(len(cycles_path)):
        # Print the current cycles we are loading
        print("Cycle {}/{}".format(c+1, len(cycles_path)),end="\r")
            
        # Load only specific passes
        if passes != []:
            # Determine passes to load
            PASSES = ["_"+str(s).zfill(3)+"_*" for s in passes]

            EXTERNAL_SUBPATH = [cycles_path[c]+"\\*"+f for f in PASSES]

            SWOT_GDR_filenames = glob.glob(EXTERNAL_SUBPATH[0])
            for i in range(len(passes)):
                if i > 0:
                    SWOT_GDR_filenames += glob.glob(EXTERNAL_SUBPATH[i])

        # Load first file to setup keys
        SWOT_GDR_filename = SWOT_GDR_filenames[0]
        GDR_file = nc.Dataset(SWOT_GDR_filename, "r")

        keys = GDR_file["data_20"].variables.keys()
        keys_data = GDR_file["data_20"]["ku"].variables.keys()
        keys_01 = GDR_file["data_01"].variables.keys()

        # Setup the passes in the dictionary from the filenames
        gdr_pass = {}
        for i in range(len(SWOT_GDR_filenames)):
            p = int(SWOT_GDR_filenames[i][49:52])
            gdr_pass[p] = gdr_pass

        # For each pass, load the file into the dict
        for i, SWOT_GDR_filename in enumerate(SWOT_GDR_filenames):
            # Print the current file
            #print("File {}/{}".format(i+1, len(SWOT_GDR_filenames)),end="\r")
            GDR_file = nc.Dataset(SWOT_GDR_filename, "r")


            # Load data for each file
            keys = GDR_file["data_20"].variables.keys()
            cyc = GDR_file.cycle_number
            pas = GDR_file.pass_number

            # Get data and load into the dict
            gdr_01data = {}
            for k in keys_01:
                gdr_01data[k] = np.interp(GDR_file["data_20"]["latitude"][:], GDR_file["data_01"]["latitude"][:], GDR_file["data_01"][k][:])
                gdr_pass[pas] = gdr_01data
                nadir[cyc] = gdr_pass

            # Get data and load into the dict
            gdr_metadata = {}
            for k in keys:
                gdr_metadata[k] = np.ma.getdata(GDR_file["data_20"][k][:])
                gdr_pass[pas].update(gdr_metadata)
                nadir[cyc] = gdr_pass
                
            # Get data and load into the dict
            gdr_data = {}
            for k in keys_data:
                gdr_data[k] = np.ma.getdata(GDR_file["data_20"]["ku"][k][:])
                gdr_pass[pas].update(gdr_data)
                nadir[cyc] = gdr_pass


            # Convert to -180 to 180 degrees longitude
            nadir[cyc][pas]["longitude"][nadir[cyc][pas]["longitude"]>180] -= 360

    # Create datetimes
    for c in nadir.keys():
        for p in nadir[c].keys():
            nadir[c][p]["datetime"] = np.array([datetime.datetime(2000,1,1)+datetime.timedelta(seconds=t) for t in nadir[c][p]["time"]])

    # Create SSH
    for c in nadir.keys():
        for p in nadir[c].keys():
            geocorr = nadir[c][p]["ocean_tide_got"] +\
                        nadir[c][p]["solid_earth_tide"] +\
                        nadir[c][p]["pole_tide"] +\
                        nadir[c][p]["dac"]
            H = nadir[c][p]["altitude"] - nadir[c][p]["range_ocean"] - nadir[c][p]["geoid"] - geocorr
            H[abs(H)>100] = np.nan
            nadir[c][p]["height"] = H

    # Pass the data
    return nadir
# ========================================================================


# ========================================================================
# Load Poseidon-3C 1Hz Nadir Altimeter data
def loadNadir1Hz(path, maxfiles=None, cycles=[], passes=[]):
    """
    Inputs:
        Path: str
        maxfiles:
    
    Outputs:
        nadir: 
    """

    # Ex: "D:\MasterThesis\SIGDR\*"
    EXTERNAL_DATA_PATH = path

    # Load filenames in path
    cycles_path = glob.glob(path)

    # Reduce cycles path to only contain certain cycles
    if cycles != []:
        cycles_path = [s for s in cycles_path if "560" in s]

    # Determine if we have a max filecount
    # If we have then reduce
    if maxfiles != None:
        cycles_path = cycles_path[:maxfiles]

    # Setup cycles for Nadir dictionary from filenames in path
    nadir = {}
    for i in range(len(cycles_path)):
        c = int(cycles_path[i][-3:])
        nadir[c] = {}

    # For each cycle, load each pass
    for c in range(len(cycles_path)):
        # Print the current cycles we are loading
        print("Cycle {}/{}".format(c+1, len(cycles_path)),end="\r")
            
        # Load only specific passes
        if passes != []:
            #PASSES = ["*_"+str(s).zfill(3)+"_*" for s in passes]
            
            # Determine passes to load
            PASSES = ["_"+str(s).zfill(3)+"_*" for s in passes]
                
            EXTERNAL_SUBPATH = [cycles_path[c]+"\\*"+f for f in PASSES]

            SWOT_GDR_filenames = glob.glob(EXTERNAL_SUBPATH[0])
            for i in range(len(passes)):
                if i > 0:
                    SWOT_GDR_filenames += glob.glob(EXTERNAL_SUBPATH[i])

        # Load first file to setup keys
        SWOT_GDR_filename = SWOT_GDR_filenames[0]
        GDR_file = nc.Dataset(SWOT_GDR_filename, "r")

        keys = GDR_file["data_01"].variables.keys()
        keys_data = GDR_file["data_01"]["ku"].variables.keys()
        keys_01 = GDR_file["data_01"].variables.keys()

        # Setup the passes in the dictionary from the filenames
        gdr_pass = {}
        for i in range(len(SWOT_GDR_filenames)):
            p = int(SWOT_GDR_filenames[i][49:52])
            gdr_pass[p] = gdr_pass

        # For each pass, load the file into the dict
        for i, SWOT_GDR_filename in enumerate(SWOT_GDR_filenames):
            # Print the current file
            #print("File {}/{}".format(i+1, len(SWOT_GDR_filenames)))
            GDR_file = nc.Dataset(SWOT_GDR_filename, "r")


            # Load data for each file
            keys = GDR_file["data_01"].variables.keys()
            cyc = GDR_file.cycle_number
            pas = GDR_file.pass_number

            # Get data and load into the dict
            gdr_metadata = {}
            for k in keys:
                gdr_metadata[k] = np.ma.getdata(GDR_file["data_01"][k][:])
                gdr_pass[pas] = gdr_metadata
                nadir[cyc] = gdr_pass
                
            # Get data and load into the dict
            gdr_data = {}
            for k in keys_data:
                gdr_data[k] = np.ma.getdata(GDR_file["data_01"]["ku"][k][:])
                gdr_pass[pas].update(gdr_data)
                nadir[cyc] = gdr_pass

            # Convert to -180 to 180 degrees longitude
            nadir[cyc][pas]["longitude"][nadir[cyc][pas]["longitude"]>180] -= 360

    # Create datetimes
    for c in nadir.keys():
        for p in nadir[c].keys():
            nadir[c][p]["datetime"] = np.array([datetime.datetime(2000,1,1)+datetime.timedelta(seconds=t) for t in nadir[c][p]["time"]])

    # Create SSH
    for c in nadir.keys():
        for p in nadir[c].keys():
            geocorr = nadir[c][p]["ocean_tide_got"] +\
                        nadir[c][p]["solid_earth_tide"] +\
                        nadir[c][p]["pole_tide"] +\
                        nadir[c][p]["dac"]
            H = nadir[c][p]["altitude"] - nadir[c][p]["range_ocean"] - nadir[c][p]["geoid"] - geocorr
            H[abs(H)>100] = np.nan
            nadir[c][p]["height"] = H

    # Pass the data
    return nadir
    # =========================================================================================


# ========================================================================
# Load KaRIn L2B 2km Altimetry data


# ========================================================================
# Load KaRIn L2A 250m Altimetry data
def loadKarinL2a(path, maxfiles=None, cycles=[], passes=[]):
    
    #ex: "D:/MasterThesis/PreBetaKaRIn_21day/unsmoothed/*"
    EXTERNAL_DATA_PATH = path
    
    # Determine passes to load
    if cycles == []:
        PASSES = ["_"+str(s).zfill(3)+"_*" for s in passes]
    else:
        PASSES = ["_"+str(c).zfill(3)+"_"+str(s).zfill(3)+"_*" for c in cycles for s in passes]
    EXTERNAL_SUBPATH = [EXTERNAL_DATA_PATH+f for f in PASSES]
        
    # Determine cycles for use and add to the list
    cycles_subpath = glob.glob(EXTERNAL_SUBPATH[0])
    for i in range(len(PASSES)):
        if i > 0:
            cycles_subpath += glob.glob(EXTERNAL_SUBPATH[i])

    # Determine if we have a max filecount
    # If we have then reduce
    if maxfiles != None:
        cycles_subpath = cycles_subpath[:maxfiles]

    # Print warning if list is empty
    if cycles_subpath == []:
        print("WARNING: no files matching criteria")

    karin_hr = {}
    for i in range(len(cycles_subpath)):
        c = int(cycles_subpath[i][71:75].replace("_",""))
        karin_hr[c] = {}

    # Sort keys in Karin
    keys = list(karin_hr.keys())
    keys.sort()
    karin_hr = {k: karin_hr[k] for k in keys}


    for i, c in enumerate(list(karin_hr.keys())):
        print("Cycle {} {}/{}".format(c, i, len(cycles_subpath)),end="\r")
        SWOT_filenames = cycles_subpath

        SWOT_filename = SWOT_filenames[0]
        file = nc.Dataset(SWOT_filename, "r")

        file.variables

        karin_pass = {}
        for i in range(len(SWOT_filenames)):
            p = int(SWOT_filenames[i][75:79].replace("_",""))
            karin_pass[p] = {}

        for i, p in enumerate(list(karin_pass.keys())):
            #print("File {}/{}".format(i+1, len(list(karin_pass.keys()))),end="\r")
            local_filename = glob.glob(EXTERNAL_DATA_PATH+"*"+f'{c:03d}'+"_"+f'{p:03d}'+"_*")
            if local_filename == []:
                karin_pass.pop(p)
                continue
            file = nc.Dataset(local_filename[0], "r")

            # Load data for each file
            keys = file["right"].variables.keys()
            cyc = file.cycle_number
            pas = file.pass_number

            # Setup sides
            for side in ["left", "right"]:
                karin_pass[pas] = {}

            # Get data
            for side in ["left", "right"]:
                data_ = {}
                for k in keys: 
                    # Load the data and remove the margins of size 4 to keep good data
                    data_[k] = np.ma.getdata(file[side][k][:])
                    if np.ndim(data_[k]) > 1: # If we have a 2d-array we remove sides
                        data_[k] = data_[k][:,4:-4]
                    karin_pass[pas][side] = data_
                    karin_hr[cyc] = karin_pass

            # Convert to -180 to 180 degrees longitude
            for side in ["left", "right"]:
                karin_hr[cyc][pas][side]["longitude"][karin_hr[cyc][pas][side]["longitude"]>180] -= 360

    # Compute custom variables
    print("Computing custom SSH variables")
    for p in karin_hr[c].keys():
        for s in karin_hr[c][p].keys():
            karin_hr[c][p][s]["ssh_ellip"] = karin_hr[c][p][s]["ssh_karin_2"]
            karin_hr[c][p][s]["ssh_mss"] = karin_hr[c][p][s]["ssh_karin_2"] - karin_hr[c][p][s]["mean_sea_surface_cnescls"]

    return karin_hr


# ========================================================================
# Load KaRIn L2HR PIXC Altimetry data




