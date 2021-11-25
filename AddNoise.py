#%%
import numpy as np
from scipy.fft import fft2, ifft2
from PIL import Image
import os

maindir = '../imagenette2/'
cats = os.listdir(maindir)

def generatenoise(img,dim):
    f = np.fft.fftn(img)                                    #fourier transform of image
    fPhase = np.angle(f)                                    #phase component of fft
    fMag = np.abs(f)                                        #magnitude component of fft
    
    rng = np.random.default_rng()                           #rng seed
    rng.shuffle(fPhase,0)                                   #shuffle phases in x 
    rng.shuffle(fPhase,1)                                   #shuffle phases in y 

    combined = np.multiply(np.abs(f), np.exp(1j * fPhase))  #recombine magnitude with shifted phases
    imgCombined = np.real(np.fft.ifftn(combined))           #inverse fft to recreate original image
    imgCombined = np.abs(imgCombined)                       #take absolute value of recombination to-
                                                            #eliminate value clipping errors in final image

    absfImg = Image.fromarray(imgCombined.astype('uint8'),'RGB') #convert phase-shifted noise array to PIL Image
    absfImg = absfImg.resize(dim)                                #resize image to given dimensions
    #absfImg.show('absf.png')   #optional show image
    #absfImg.save('f2.png')     #optional save image
    

    return absfImg
    
   
## Testing Code ## 
testImg = Image.open(maindir + 'n03417042_118.JPEG')
#testImg.show()
testImg = np.array(testImg)
dim = (1000,1000)

generatenoise(testImg, dim) # TEST
