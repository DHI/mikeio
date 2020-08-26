import numpy as np

def safe_length(input_list):
    """
    Get the length of a Python or C# list.

    Usage:
       safe_length(input_list)

    input_list : Python or C# list

    Return:
        int
           Integer giving the length of the input list.
    """

    try:
        n = input_list.Count
    except:
        n = len(input_list)

    return n

def min_horizontal_dist_meters(coords, targets, is_geo=False):
    xe = coords[:,0]
    ye = coords[:,1]
    n = len(xe)
    d  = np.zeros(n)
    for j in range(n):
        d1 = dist_in_meters(targets, [xe[j], ye[j]], is_geo=is_geo)
        d[j] = d1.min()
    return d

def dist_in_meters(coords, pt, is_geo=False):
    xe = coords[:,0]
    ye = coords[:,1]
    xp = pt[0]
    yp = pt[1]    
    if is_geo:        
        d = _get_dist_geo(xe, ye, xp, yp)
    else:   
        d = np.sqrt(np.square(xe - xp) + np.square(ye - yp))
    return d

def _get_dist_geo(lon, lat, lon1, lat1):
    # assuming input in degrees!
    R = 6371e3 # Earth radius in metres
    dlon = np.deg2rad(lon1 - lon)
    if(np.any(dlon>np.pi)): 
        dlon = dlon - 2*np.pi
    if(np.any(dlon<-np.pi)): 
        dlon = dlon + 2*np.pi    
    dlat = np.deg2rad(lat1 - lat)
    x = dlon*np.cos(np.deg2rad((lat+lat1)/2))
    y = dlat
    d = R * np.sqrt(np.square(x) + np.square(y) )
    return d