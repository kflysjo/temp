import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_s_csv(filename, parameter, angle_first=False):
    df = pd.read_csv(filename)

    mag = f"{parameter}_mag"
    angle = f"{parameter}_angle_deg"
    parameter_columns = df.columns[:-3].tolist()

    if angle_first:
        df.columns = parameter_columns + ["freq_GHz", angle, mag]
    else:
        df.columns = parameter_columns + ["freq_GHz", mag, angle]

    df[parameter] = df[mag] * np.exp(
        1j * np.deg2rad(df[angle])
    )

    return df


def s_to_abcd(S11, S21, S12, S22, Zref):

    denom = 2 * S21
    T = np.empty((len(S11), 2, 2), dtype=complex)

    T[:, 0, 0] = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    T[:, 0, 1] = Zref * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    T[:, 1, 0] = ((1 - S11) * (1 - S22) - S12 * S21) / (Zref * denom)
    T[:, 1, 1] = ((1 - S11) * (1 + S22) + S12 * S21) / denom

    return T


def substrate_abcd(frequency, thickness, eps_r):
    Zd = 376.730313668 / np.sqrt(eps_r)
    theta = (
        2 * np.pi * frequency * np.sqrt(eps_r)
        * thickness / 299792458
    )

    T = np.zeros((len(frequency), 2, 2), dtype=complex)
    T[:, 0, 0] = np.cos(theta)
    T[:, 1, 1] = np.cos(theta)
    T[:, 0, 1] = 1j * Zd * np.sin(theta)
    T[:, 1, 0] = 1j * np.sin(theta) / Zd

    return T



Zref = 376.730313668
substrate_eps_r = 3.8
air_thickness_left = 1e-3
air_thickness_right = 1e-3
abcd_data = {}
metal_abcd = {}

S11 = read_s_csv("S11_magang_test.csv", "S11")
S21 = read_s_csv("S21_magang_test.csv", "S21", angle_first=False)

sweep_columns = S11.columns[:-4].tolist()

if sweep_columns != S21.columns[:-4].tolist():
    raise ValueError("S11 och S21 har olika parameterkolumner.")

df = S11.merge(
    S21,
    on=sweep_columns + ["freq_GHz"],
    how="outer",
    validate="one_to_one",
    indicator=True
)

if (df["_merge"] != "both").any():
    raise ValueError("S11 och S21 har olika parameter- eller frekvenspunkter.")

df = df.drop(columns="_merge")
df["freq_Hz"] = df["freq_GHz"] * 1e9
patch_data = {}

for _, group in df.groupby(sweep_columns):
    label = ", ".join(
        f"{column} = {group[column].iloc[0]:g}"
        for column in sweep_columns
    )

    patch_data[label] = (
        group.drop(columns=sweep_columns + ["freq_GHz"])
             .set_index("freq_Hz")
             .sort_index()
    )


for label, data in patch_data.items():
    frequency = data.index.to_numpy()
    S11 = data["S11"].to_numpy()
    S21 = data["S21"].to_numpy()


    T_total = s_to_abcd(
        S11=S11,
        S21=S21,
        S12=S21,
        S22=S11,
        Zref=Zref
    )


    T_left = substrate_abcd(frequency, air_thickness_left, 1.0)
    T_right = substrate_abcd(frequency, air_thickness_right, 1.0)

    T_metal = (
        np.linalg.inv(T_left)
        @ T_total
        @ np.linalg.inv(T_right)
    )


    Y_metal = T_metal[:, 1, 0]
    data["C_eff_F"] = Y_metal.imag / (2 * np.pi * frequency)

    # Spara mellanresultaten för senare analys
    abcd_data[label] = T_total
    metal_abcd[label] = T_metal

fig, ax = plt.subplots(figsize=(9, 5))

for ratio, data in patch_data.items():
    ax.plot(
        data.index.to_numpy() / 1e9,
        data["C_eff_F"] * 1e15,
        label=ratio
    )

ax.set_xlabel("Frekvens [GHz]")
ax.set_ylabel("Effektiv kapacitans [fF]")
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()


freq_of_interest = 86e9

for label, data in patch_data.items():
    i = abs(data.index.to_numpy() - freq_of_interest).argmin()
    freq = data.index[i]
    capacitance = data["C_eff_F"].iloc[i] * 1e15

    print(f"{label}: {capacitance:.3f} fF vid {freq / 1e9:.3f} GHz")
