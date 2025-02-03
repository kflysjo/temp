import io_chalm
import numpy as np
import matplotlib.pyplot as plt


def create_design_matrix(data):
    A = data["A"].values
    Z = data["Z"].values
    N = data["N"].values

    # Paritetstermer: δ = (+1 för jämn-jämn, -1 för udda-udda, 0 annars)
    delta = np.where((Z % 2 == 0) & (N % 2 == 0), 1, \
                     np.where((Z % 2 != 0) & (N % 2 != 0), -1, 0))

    # Bygg designmatrisen X
    X = np.column_stack([
        A,                    # Volymterm (aV)
        -A**(2/3),             # Ytterterm (aS)
        -Z**2 / A**(1/3),      # Coulombterm (aC)
        #-Z*(Z-1) / A**(1/3),      # Coulombterm (aC)

        -(Z - A/2)**2 / A,     # Asymmetriterm (aA)
        delta / A**(1/2)      # Paritetsterm (aP)
    ])

    return X

def B_SEMF(A, Z):
    """
    Beräknar bindningsenergin B för en kärna med massantal A och protontal Z
    enligt den semi-empiriska massformeln (SEMF).
    
    Parametrar:
    A -- Massantal (totalt antal nukleoner)
    Z -- Protontal (antal protoner)
    
    Returvärde:
    B -- Bindningsenergi i MeV
    """
    
    # Semi-empiriska massformelns koefficienter (exempelvärden, bör kalibreras från regression)
    a_v = 15.49      # Volymterm (MeV)
    a_s = 16.79      # Ytterterm (MeV)
    a_c = 0.70     # Coulombterm (MeV)
    a_a = 91.61     # Asymmetriterm (MeV)
    a_p = 12.17      # Paritetsterm (MeV)
    
    # Beräkna neutronantal
    N = A - Z
    
    # Paritetsterm δ: (+1 för jämn-jämn, -1 för udda-udda, 0 annars)
    if (Z % 2 == 0) and (N % 2 == 0):
        delta = 1  # Jämn-jämn kärna
    elif (Z % 2 == 1) and (N % 2 == 1):
        delta = -1  # Udda-udda kärna
    else:
        delta = 0  # Jämn-udda eller udda-jämn kärna

    # Beräkna bindningsenergin enligt SEMF
    B = (a_v * A 
         - a_s * A**(2/3) 
         - a_c * (Z**2 / A**(1/3)) 
         - a_a * (Z - A/2)**2 / A
         + a_p * delta / A**(1/2))

    return B



# skapa data frame osv
file_path = "ame2016.txt"
df = io_chalm.load_ame_data(file_path)
X = create_design_matrix(df)
B = df["B"].values
y = B
BA = df["BA"].values
N = df["N"].values
A = df["A"].values
Z = df["Z"].values


#Lös normalekvationen
a_vec = np.linalg.inv(X.T @ X) @ X.T @ y

# Beräkna modellprediktioner
y_model = X @ a_vec

# Beräkna residualerna
residuals = y - y_model


# Beräkna covarians och correlationsmatris
cov_a = (residuals.T @ residuals) * np.linalg.inv(X.T @ X) / (len(N)-len(a_vec))
corr_matrix = np.zeros_like(cov_a)
for i in range(len(a_vec)):
    for j in range(len(a_vec)):
        corr_matrix[i, j] = cov_a[i, j] / (np.sqrt(cov_a[i,i]) * np.sqrt(cov_a[j,j]))


print(f"Parametervärden: {a_vec}")
print("Korrelationsmatris för parametrarna:")
print(corr_matrix)

'''

# Plotta residualerna mot neutronantal N (begränsat till intervallet)
plt.figure(figsize=(10, 4))
plt.plot(N, residuals, color='blue', alpha=0.7, label="Residualer")

# Markera magiska neutronantal med vertikala linjer
magic_numbers = [28, 50, 82, 126]
for i, magic in enumerate(magic_numbers):
    if 20 <= magic <= 140:  # Markera bara om inom intervallet
        plt.axvline(x=magic, color='red', linestyle='--', alpha=0.6, label=f'N = {magic}')

# Anpassa plotten
plt.title("Residualer: Skillnad mellan experimentella och beräknade bindningsenergier (N mellan 20 och 140)")
plt.xlabel("Neutrontal (N)")
plt.ylabel(r"Residualer (MeV/c$^2$)")
plt.legend(loc='upper right')
plt.xlim(4, 140)
plt.grid(alpha=0.3)
plt.show()

# Plotta residualerna mot neutronantal Z (begränsat till intervallet)
plt.figure(figsize=(10, 4))
plt.plot(Z, residuals, color='blue', alpha=0.7, label="Residualer")

# Markera magiska neutronantal med vertikala linjer
magic_numbers = [28, 50, 82, 126]
for i, magic in enumerate(magic_numbers):
    if 20 <= magic <= 140:  # Markera bara om inom intervallet
        plt.axvline(x=magic, color='red', linestyle='--', alpha=0.6, label=f'Z = {magic}')

# Anpassa plotten
plt.title("Residualer: Skillnad mellan experimentella och beräknade bindningsenergier (Z mellan 20 och 140)")
plt.xlabel("Neutrontal (Z)")
plt.ylabel(r"Residualer (MeV/c$^2$)")
plt.legend(loc='upper right')
plt.xlim(4, 140)
plt.grid(alpha=0.3)
plt.show()

'''


# Energiparabel för A = 102
A_102 = 102
mask_102 = (df["A"] == A_102)
Z_filtered = Z[mask_102]
mask_Z_even = (Z_filtered % 2 == 0)
mask_Z_odd = (Z_filtered % 2 == 1)

Z_filtered_even = Z_filtered[mask_Z_even]
Z_filtered_odd = Z_filtered[mask_Z_odd]

B_vec_teori_even = -np.array([B_SEMF(A_102, Z) for Z in Z_filtered_even])
B_vec_teori_odd = -np.array([B_SEMF(A_102, Z) for Z in Z_filtered_odd])

# Energiparabel för A = 111
A_111 = 111
mask_111 = (df["A"] == A_111)
Z_filtered_111 = Z[mask_111]

B_vec_teori_111 = -np.array([B_SEMF(A_111, Z) for Z in Z_filtered_111])


B_exp_filtered = -B[mask_102]  # Negativ bindningsenergi
B_exp_filtered_111 = -B[mask_111]  # Negativ bindningsenergi

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]

ax.plot(Z_filtered_even, B_vec_teori_even, color="blue", label="jämn-jämn (teori)", marker="o")
ax.plot(Z_filtered_odd, B_vec_teori_odd, color="green", label="udda-udda (teori)", marker="s")
for i, Z_val in enumerate(Z_filtered):  # Loop med indexering
    if (Z_val % 2 == 0) & ((A_102 - Z_val) % 2 == 0):  # Jämn-jämn
        ax.scatter(Z_val, B_exp_filtered[i], color="blue", label="jämn-jämn (exp)" if i == 0 else "", marker="o")

    if (Z_val % 2 == 1) & ((A_102 - Z_val) % 2 == 1):  # Udda-udda
        ax.scatter(Z_val, B_exp_filtered[i], color="green", label="udda-udda (exp)" if i == 1 else "", marker="s")


# Anpassa plotten
ax.set_title("Energiparabel A = 102")
ax.set_xlabel("Protontal (Z)")
ax.set_ylabel("Grundenergitillstånd (MeV)")
ax.legend(loc="upper center")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(Z_filtered_111, B_vec_teori_111, color="blue", label="Teori", marker="o")
ax.scatter(Z_filtered_111, B_exp_filtered_111, color="red", label="Experiment", marker="s")

# Anpassa plotten
ax.set_title("Energiparabel A = 111")
ax.set_xlabel("Protontal (Z)")
ax.set_ylabel("Grundenergitillstånd (MeV)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout
plt.show()
