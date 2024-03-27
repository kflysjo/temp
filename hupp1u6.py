import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

def fft2c(x):
    '''
    2D Fourier transform
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

def ifft2c(x):
    '''
    2D inverse Fourier transform
    '''
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x)))


def PAS(E1, L, N, a, lam0, n):
    '''
    Funktion för att propagera E1 sträckan L genom PAS
    '''
    # Varje sampelpunkt i k-planet motsvarar en plan våg med en viss riktning [kx,ky,kz]
    delta_k = 2 * np.pi / (N * a)                                           # Samplingsavstånd i k-planet (Vill ha 2 *pi / (N * a))

    kx      = np.arange(-(N/2)*delta_k, (N/2)*delta_k, delta_k) # Vektor med samplingspunkter i kx-led
    ky      = kx                                                # och ky-led
    KX, KY  = np.meshgrid(kx,ky)                        # k-vektorns x- resp y-komponent i varje
                                                        # sampelpunkt i k-planet

    k = 2*np.pi*n / lam0                                # k-vektorns längd (skalär) för en plan våg i ett material med brytningsindex n
    KZ = np.sqrt(k**2-KX**2-KY**2, dtype=complex)       # k-vektorns z-komponent i varje sampelpunkt.
    fasfaktor_propagation = np.exp(1j*KZ*L)             # Faktor för varje sampelpunkt i k-planet, multas med för att propagera sträckan L i z-led

    A  = (a / (2*np.pi))**2 *fft2c(E1)                 # Planvågsspektrum i Plan 1
    B  = A*fasfaktor_propagation                        # Planvågsspektrum i Plan 2 (Planvågsspektrum i Plan 1 multat med fasfaktorn för propagation)
    E2 = delta_k**2 * N**2 *ifft2c(B)

    return E2


N               = 2**10                # NxN är antalet samplade punkter (rekommenderad storlek N=1024)
sidlaengd_Plan1 = 4e-3                  # Det samplade områdets storlek (i x- eller y-led) i Plan 1 (rekommenderad storlek 4 mm)
a               = sidlaengd_Plan1/N     # Samplingsavstånd i Plan 1 (och Plan 2 eftersom vi använder PAS)
lambda_noll     = 633e-9                    # Vakuumvåglängd för rött ljus från en HeNe-laser
n_medium        = 1                         # Brytningsindex för medium mellan Plan 1 och 2
k               = 2*np.pi*n_medium / lambda_noll       #!!!                # K-vektorns längd
L               = 100e-3                # Propagationssträcka (dvs avstånd mellan Plan 1 och 2)


### Definera koordianter i plan 1 ###
x = np.arange(-(N/2)*a, (N/2)*a, a)     # Vektor med sampelpositioner i x-led
y = x                                   # och y-led

X, Y = np.meshgrid(x, y)                # Koordinatmatriser med x- och y-värdet i varje sampelposition
R    = np.sqrt(X**2 + Y**2)             # Avståndet till origo för varje sampelpunkt

### Definera lins och cirkulär aperatur ###
f_lins = 100e-3                         # Fokallängd på linsen före Plan 1 100e-3
T_lins = np.exp(-1j*k*R**2/(2*f_lins))  # Transmissionsfunktion för en lins (linsen är TOK)

D_aperture = 2e-3                       # Diameter för apertur
T_aperture = R < (D_aperture/2)         # Transmissionsfunktion för en cirkulär apertur ("pupill")


### Definera fält i plan 1 ###
omega1      = 1e-3                      # 1/e2-radie (för intensiteten, dvs 1/e-radie för amplituden) för infallande Gaussiskt fält
E1_in_gauss = np.exp(-R**2/omega1**2)   # Infallande fält: Gaussiskt med plana vågfronter och normalinfall (dvs konstant fas, här=0)
E1_in_konst = np.ones(X.shape)          # Infallande fält: Konstant i hela plan 1. np.ones(X.shape) ger en matris fylld med ettor som har samma storlek som X
E1_punkt_4m = np.exp(-1j*k*np.sqrt(R**2 +4**2))/np.sqrt(R**2 +4**2)  # Infallande fält från punktkälla, 4 meter bort



E1_gauss    = E1_in_gauss*T_lins        # Fältet i Plan 1 (precis efter linsen) för gaussisk stråle
E1_cirkular = E1_in_konst* T_lins * T_aperture     # Fältet i Plan 1 (precis efter linsen) för konstant fält som passerat genom cirkulär apertur *** Ej klar
E1_punkt    = E1_punkt_4m * T_lins * -0.11189169331042044+0.22356262873773608j

phases = np.random.uniform(0, 2*np.pi, (1024, 1024))
amplitude = 1  
E1_randon_phase = amplitude * np.exp(1j * phases)

E1          = E1_punkt               # Välj fall!

#Det ofarliga meddelandet

DOE = np.flip(io.loadmat('C:\\Users\\karlf\\Desktop\\Optik\\HUPP1\\T_DOE_gen2.mat'))
T_DOE = DOE[list(DOE.keys())[-1]]

f_eye = 4
T_eye = np.exp(-1j*k*R**2/(2*f_eye))  # Transmissionsfunktion för ögat (ögat är TOK)

E_P = E1 * T_DOE * T_eye

L = 20e-3
cornea = PAS(E_P, L, N, a, lambda_noll, n_medium)         # Propagera med vår PAS funktion

I2      = np.abs(cornea)**2
I2_norm = np.log(I2/np.max(I2))  # Log av den normaliserade intensiteten i plan 2
x_mm = x*1e3
y_mm = y*1e3

plt.figure(1)
image = plt.imshow(I2_norm, extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()])
plt.colorbar(image)
plt.show()



#Det farliga meddelandet dvl=20mm

DOE = np.flip(io.loadmat('C:\\Users\\karlf\\Desktop\\Optik\\HUPP1\\T_DOE_gen2.mat'))
T_DOE = DOE[list(DOE.keys())[-1]]

f_eye = 4
T_eye = np.exp(-1j*k*R**2/(2*f_eye))  # Transmissionsfunktion för ögat (ögat är TOK)
f_dvl = 20e-3
T_dvl = np.exp(-1j*k*R**2/(2*f_dvl))  # Transmissionsfunktion för ögat (ögat är TOK)

E_P = E1 * T_DOE * T_eye * T_dvl

L = 20e-3
cornea = PAS(E_P, L, N, a, lambda_noll, n_medium)         # Propagera med vår PAS funktion

I2      = np.abs(cornea)**2
I2_norm = np.log(I2/np.max(I2))  # Log av den normaliserade intensiteten i plan 2
x_mm = x*1e3
y_mm = y*1e3

plt.figure(2)
image = plt.imshow(I2_norm, extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()])
plt.colorbar(image)
plt.show()



#Det farliga meddelandet dvl=23mm

DOE = np.flip(io.loadmat('C:\\Users\\karlf\\Desktop\\Optik\\HUPP1\\T_DOE_gen2.mat'))
T_DOE = DOE[list(DOE.keys())[-1]]

f_eye = 4
T_eye = np.exp(-1j*k*R**2/(2*f_eye))  # Transmissionsfunktion för ögat (ögat är TOK)
f_dvl = 23e-3
T_dvl = np.exp(-1j*k*R**2/(2*f_dvl))  # Transmissionsfunktion för ögat (ögat är TOK)

E_P = E1_in_konst * T_DOE * T_eye * T_dvl

L = 20e-3
cornea = PAS(E_P, L, N, a, lambda_noll, n_medium)         # Propagera med vår PAS funktion

I2      = np.abs(cornea)**2
I2_norm = np.log(I2/np.max(I2))  # Log av den normaliserade intensiteten i plan 2
x_mm = x*1e3
y_mm = y*1e3

plt.figure(3)
image = plt.imshow(I2_norm, extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()])
plt.colorbar(image)
plt.show()
