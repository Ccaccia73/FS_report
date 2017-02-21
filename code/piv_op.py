import numpy as np
from skimage.feature import register_translation


def min_max_filter():
	pass
	

def corr_wind(im_a,im_b,window_size, max_ext,y0,x0):
    
    corr_mat = np.zeros((2*max_ext+1,2*max_ext+1))
        
    for j in range(-max_ext,max_ext+1):
        for i in range(-max_ext,max_ext+1):
            corr_mat[j+max_ext,i+max_ext] = np.sum(np.power(im_a[y0+j:y0+j+window_size,x0+i:x0+i+window_size] - 
                                                            im_b[y0:y0+window_size,x0:x0+window_size], 2))
    return corr_mat


def normalized_corr_wind(im_a,im_b,window_size, max_ext,y0,x0):
    
    corr_mat = np.zeros((2*max_ext+1,2*max_ext+1))
    
    for j in range(-max_ext,max_ext+1):
        for i in range(-max_ext,max_ext+1):
            imgA = im_a[y0+j:y0+j+window_size,x0+i:x0+i+window_size]
            imgB = im_b[y0:y0+window_size,x0:x0+window_size]
            
            Amean = np.mean(imgA)
            Bmean = np.mean(imgB)
            
            Astd = np.std(imgA)
            Bstd = np.std(imgB)
            
            corr_mat[j+max_ext,i+max_ext] = np.sum((imgA-Amean)*(imgB-Bmean)/((2*window_size+1)**2*Astd*Bstd))
            
    return corr_mat

def corr_wind_SK(im_a,im_b,window_size, max_ext,y0,x0):
    shift, error, diffphase = register_translation(im_a[y0:y0+window_size,x0:x0+window_size], 
                                                   im_b[y0:y0+window_size,x0:x0+window_size],100)
    
    #print error
    
    return shift



def find_best_corr(corr_matrix, find_max=False):
    if find_max:
        i,j = np.unravel_index(corr_matrix.argmax(), corr_matrix.shape)
    else:
        i,j = np.unravel_index(corr_matrix.argmin(), corr_matrix.shape)
    return i,j


def subpixel(corr_matrix,dy,dx):
    #print corr_matrix.shape
    if (dy > 0 and dy < corr_matrix.shape[0]-1) and (dx > 0 and dx < corr_matrix.shape[1]-1):
        phi_0 = corr_matrix[dy,dx]
        
        phi_1y = corr_matrix[dy+1,dx]
        phi_1x = corr_matrix[dy,dx+1]
        
        phi_m1y = corr_matrix[dy-1,dx]
        phi_m1x = corr_matrix[dy,dx-1]
        
        eps_x = 0.5*np.log(phi_m1x/phi_1x)/np.log(phi_m1x/(phi_1x*phi_0**2))
        eps_y = 0.5*np.log(phi_m1y/phi_1y)/np.log(phi_m1y/(phi_1y*phi_0**2))
    
    else:
        eps_y = 0.0
        eps_x = 0.0
    
    return eps_y, eps_x

def point_inside_polygon(x,y,xpoly, ypoly):
    
    poly = zip(xpoly,ypoly)
    
    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside