import numpy as np
import pandas as pd


def create_design_mat(df, tot_width=False):
    """
    Completes the preprocessing of DataFrame to create the design matrix for telescope position problem

            output:
            design_mat
            pinv
            y
            beta

    """
    # ea. row must contain (x, y, z) for ea. telescope (except E1, the zero point)
    # and all 30 POP settings for ea. telescope
    telescopes = ["E1", "W2", "W1", "S2", "S1", "E2"]
    theta = df.elevation
    phi = df.azimuth
    S = np.array(
        [np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), np.sin(theta)]
    ).T
    k = 0

    tot_unique_pops = []

    width = 15

    for telescope in telescopes:

        pop_tel_1 = df[df.tel_1 == telescope].pop_1
        pop_tel_2 = df[df.tel_2 == telescope].pop_2
        tel_1_unique_pops = np.unique(pop_tel_2)
        tel_2_unique_pops = np.unique(pop_tel_1)
        tel_unique_pops = np.union1d(tel_1_unique_pops, tel_2_unique_pops)
        tot_unique_pops.append(tel_unique_pops)
        width += len(tel_unique_pops)

    if tot_width:
        width = 195

    design_mat = np.zeros((len(df), width))

    for i, telescope in enumerate(telescopes):

        # for ea. new telescope must jump 3 places for (x, y, z)

        if telescope != "E1":  # keep 'E1' as the zero point

            design_mat[:, 3 * i - 3 : 3 + 3 * i - 3] += S * np.where(
                df["tel_1"] == telescope, 1, 0
            ).reshape(-1, 1)
            design_mat[:, 3 * i - 3 : 3 + 3 * i - 3] -= S * np.where(
                df["tel_2"] == telescope, 1, 0
            ).reshape(-1, 1)

        for pop in tot_unique_pops[i]:

            # not working because E1 isn't in the list of telescopes
            design_mat[:, 15 + k] += np.where(
                (df["pop_1"] == pop) & (df["tel_1"] == telescope), 1, 0
            )  # add when it is tel_1
            design_mat[:, 15 + k] -= np.where(
                (df["pop_2"] == pop) & (df["tel_2"] == telescope), 1, 0
            )  # subtract when it is the tel_2

            k += 1

    y = df["cart_2"].values - df["cart_1"].values

    pinv = np.linalg.pinv(design_mat)

    beta = pinv @ y

    return design_mat, pinv, y, beta


def remove_outliers(y, beta, design_mat, df, n_sigma=2):

    # find residuals
    residual = y - design_mat @ beta

    # set threshold via standard deviation
    threshold = n_sigma * np.std(residual)

    # set index value of outliers to 1 and remove these
    df_new = df[np.where(abs(residual) > threshold, 1, 0) != 1]

    return df_new

def load_and_clean(date):
    
    df = pd.read_csv(f'data/{date}.csv')

    design_mat, pinv, y, beta = create_design_mat(df)

    df = remove_outliers(y, beta, design_mat, df, 1)

    design_mat, pinv, y, beta = create_design_mat(df)
    
    return design_mat, pinv, y, beta


def svd_uncertainty(design_mat, y, beta):
    
    residuals = y - (design_mat @ beta)
    
    sigma_d_squared = np.sum((residuals)**2) / (len(y) - len(beta))
    
    _, w, Vt = np.linalg.svd(design_mat)
    
    eps = np.finfo(type(w[0])).eps
    N = len(y)

    # w is given as a vector in ascending order
    small = w[0] * N * eps

    w = w[w > small]
    
    # remove eigenvetors orresponding to the small singular values
    Vt = Vt[:, :len(w)]
    Vt.shape
    
    sigma = np.sqrt(np.sum((Vt/w)**2, axis=1) * sigma_d_squared)
    
    return sigma


actual_locations_2005 = np.array([
    #[0,0,0], # E1
    [194.451, 106.618, - 6.318], # W2
    [300.442, 89.639, 4.954], # W1
    [131.120, 272.382, -6.508], # S2
    [125.371, 305.963, -5.865], # S1
    [54.970, 36.246, -3.077], # E2
])