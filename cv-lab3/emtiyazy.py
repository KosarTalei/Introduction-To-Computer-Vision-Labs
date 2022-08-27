


h=hist(I)
hist=np.zeros(256)
hist[0]=h[0]
for c in np.arange(1,256):
    hist[c]=h[c]+hist[c-1]
    
ny,nx=I.shape
for i in np.arange(nx):
    for j in np.arange(ny):
        J=hist[I]
    
