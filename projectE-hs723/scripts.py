import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def create_design_mat(df):
    """

    Creates design matrix from preprocessed dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with measurements


    Returns
    -------
    design_mat: numpy.ndarray
        design matrix for the dataframe
    pinv: numpy.ndarray
        Moore-Penrose pseudoinverse found using `np.linalg.pinv`
    y: numpy.ndarray
        Output vector i.e. lists of difference in carts
    beta: numpy.ndarray
        model parameters vector found using `pinv @ y`
        baseline components zero'd on E1 telescope
        15 baselines W2, W1, S2, S1, E2 followed by POP settings

    """
    # ea. row must contain (x, y, z) for ea. telescope (except E1, the zero point)
    # and all 30 POP settings for ea. telescope
    telescopes = ["E1", "W2", "W1", "S2", "S1", "E2"]

    # find the star vectors
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


def svd_uncertainty(design_mat, y, beta):
    """

    Calculate uncertainty derived from singular value decomposition removing zero singular value in the process

    Parameters
    ----------
    design_mat: numpy.ndarray
        design matrix for the dataframe
    pinv: numpy.ndarray
        Moore-Penrose pseudoinverse found using `np.linalg.pinv`
    beta: numpy.ndarray
        model parameters vector found using `pinv @ y`
        baseline components zero'd on E1 telescope
        15 baselines W2, W1, S2, S1, E2 followed by POP settings

    Returns
    -------
    sigma: numpy.ndarray
        array of standard deviations for the model parameters

    """

    residuals = y - (design_mat @ beta)

    sigma_d_squared = np.sum((residuals) ** 2) / (len(y) - len(beta))

    _, w, Vt = np.linalg.svd(design_mat)

    # find the machine precision of the data type being used
    eps = np.finfo(type(w[0])).eps
    N = len(y)

    # this gives the cutoff after which point the singular values can be considered zero
    small = w[0] * N * eps

    w = w[w > small]

    # remove eigenvetors orresponding to the small singular values
    Vt = Vt[:, : len(w)]

    sigma = np.sqrt(np.sum((Vt / w) ** 2, axis=1) * sigma_d_squared)

    # ensure that error is atleast as large as the POP drift
    return np.maximum(sigma, 0.0006807973948568247)


def pre_process(file_name):
    """

    processes by adding in extra datetime columns, swapping the elevation and azimuth columns where necessary and converting the elevation and azimuth to radians. After this the file is saved back to where it came from.

    Parameters
    ----------
    file_name: str
        where the file is located relative to current directory

    Returns
    ---------
    None
    """

    # load the csv into a DataFrame
    df = pd.read_csv(f"data/{file_name}.csv")

    # create seperate columns for year, month, dat and time
    dates = pd.to_datetime(df.utc)
    df["year"] = [date.year for date in dates]
    df["month"] = [date.month for date in dates]
    df["day"] = [date.day for date in dates]
    df["month"] = [date.month for date in dates]
    df["dayofyear"] = [date.day_of_year for date in dates]

    # need to swap the elevation and azimuth for the 2019 date sets
    if file_name[:4] == "2019":
        df = df.rename(columns={"elevation": "azimuth", "azimuth": "elevation"})

    # change to radians
    df["azimuth"] = df["azimuth"] * 2 * np.pi / 360
    df["elevation"] = df["elevation"] * 2 * np.pi / 360

    df.to_csv(f"data/{file_name}.csv", index=False)


def remove_outliers(df, n_sigma=2):
    """

    Removes outliers residual of the model +- n_sigma standard deviations from zero

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to be cleaned
    n_sigma:
        number of std devs where the cutoff stars (default=2)

    Returns
    -------
    df_new: pandas.DataFrame
        The dataframe with the outliers removed

    """

    design_mat, pinv, y, beta = create_design_mat(df)

    # find residual vector
    residual = y - design_mat @ beta

    # set threshold via standard deviation
    threshold = n_sigma * np.std(residual)

    # set index value of outliers to 1 and remove these
    df_new = df[np.where(abs(residual) > threshold, 1, 0) != 1]

    return df_new


def load_and_clean(date):
    """

    For a given data, load in the data, remove outliers and refind the model parameters

    Parameters
    ----------
    date: str
        date to get and clean data for

    Returns
    -------
    design_mat: numpy.ndarray
        design matrix for the dataframe
    pinv: numpy.ndarray
        Moore-Penrose pseudoinverse found using `np.linalg.pinv`
    y: numpy.ndarray
        Output vector i.e. lists of difference in carts
    beta: numpy.ndarray
        model parameters vector found using `pinv @ y`
        baseline components zero'd on E1 telescope
        15 baselines W2, W1, S2, S1, E2 followed by POP settings
    """

    df = pd.read_csv(f"data/{date}.csv")

    design_mat, pinv, y, beta = create_design_mat(df)

    df = remove_outliers(df, 1)

    design_mat, pinv, y, beta = create_design_mat(df)

    return design_mat, pinv, y, beta


def svd_uncertainty(design_mat, y, beta):
    """

    Calculate uncertainty derived from singular value decomposition removing zero singular value in the process

    Parameters
    ----------
    design_mat: numpy.ndarray
        design matrix for the dataframe
    pinv: numpy.ndarray
        Moore-Penrose pseudoinverse found using `np.linalg.pinv`
    beta: numpy.ndarray
        model parameters vector found using `pinv @ y`
        baseline components zero'd on E1 telescope
        15 baselines W2, W1, S2, S1, E2 followed by POP settings

    Returns
    -------
    sigma: numpy.ndarray
        array of standard deviations for the model parameters

    """

    residuals = y - (design_mat @ beta)

    sigma_d_squared = np.sum((residuals) ** 2) / (len(y) - len(beta))

    _, w, Vt = np.linalg.svd(design_mat)

    # find the machine precision of the data type being used
    eps = np.finfo(type(w[0])).eps
    N = len(y)

    # this gives the cutoff after which point the singular values can be considered zero
    small = w[0] * N * eps

    w = w[w > small]

    # remove eigenvetors orresponding to the small singular values
    Vt = Vt[:, : len(w)]

    sigma = np.sqrt(np.sum((Vt / w) ** 2, axis=1) * sigma_d_squared)

    return np.maximum(sigma, 0.0006807973948568247)


def S_vector(theta, phi):
    """

    Converts the horizontal coordinates given into Cartesian vectors for the stars

    Parameters
    ----------
    theta: numpy.ndarray
        list of elevation values
    phi: numpy.ndarray
        list of azimuth values

    Returns
    -------
    S: numpy.ndarray
        (n, 3) array of star vectors
    """

    return np.array(
        [np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), np.sin(theta)]
    ).T


def plot_graph(df, day_of_year=-1):
    """

    Creates design matrix from preprocessed dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with measurements
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string
    """

    if day_of_year != -1:
        df = df[df.dayofyear == day_of_year]

    df["telpop_1"] = df.tel_1 + df.pop_1
    df["telpop_2"] = df.tel_2 + df.pop_2

    graph_df = (
        df.groupby(["telpop_1", "telpop_2"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    G = nx.from_pandas_edgelist(graph_df, source="telpop_1", target="telpop_2")

    nx.draw(G, with_labels=True)
    plt.show()


def find_good_days(df, bin_size=1, error_threshold=0.1):
    """

        Finds the days/set of days which are provide resasonable estimates for the positions of the telescopes.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with measurements
    bin_size :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    good_days: list
        list of good days found for the dataframe under certain conditions
    """

    good_days = []

    for day in range(1, 367):
        # sets of three days where the max error from actual locations less than 0.1m
        df_temp = df[
            (day - (bin_size / 2) - 1 / 2 < df.dayofyear)
            & (df.dayofyear < day + (bin_size / 2) + 1 / 2)
        ]

        design_mat, pinv, y, beta = create_design_mat(df_temp)

        beta = beta[:15].reshape(5, 3)

        if np.max(abs(actual_locations_2005 - beta)) < error_threshold:
            good_days.append(day)

    return good_days


def pre_process(file_name):
    """

    processes by adding in extra datetime columns, swapping the elevation and azimuth columns where necessary and converting the elevation and azimuth to radians. After this the file is saved back to where it came from.

    Parameters
    ----------
    file_name: str
        where the file is located relative to current directory

    Returns
    ---------
    None
    """

    # load the csv into a DataFrame
    df = pd.read_csv(f"data/{file_name}.csv")

    # create seperate columns for year, month, dat and time
    dates = pd.to_datetime(df.utc)
    df["year"] = [date.year for date in dates]
    df["month"] = [date.month for date in dates]
    df["day"] = [date.day for date in dates]
    df["month"] = [date.month for date in dates]
    df["dayofyear"] = [date.day_of_year for date in dates]

    # need to swap the elevation and azimuth for the 2019 date sets
    if file_name[:4] == "2019":
        df = df.rename(columns={"elevation": "azimuth", "azimuth": "elevation"})

    # change to radians
    df["azimuth"] = df["azimuth"] * 2 * np.pi / 360
    df["elevation"] = df["elevation"] * 2 * np.pi / 360

    df.to_csv(f"data/{file_name}.csv", index=False)


# The locations of the telescopes as per the original paper in 2005
actual_locations_2005 = np.array(
    [
        # [0,0,0], # E1
        [194.451, 106.618, -6.318],  # W2
        [300.442, 89.639, 4.954],  # W1
        [131.120, 272.382, -6.508],  # S2
        [125.371, 305.963, -5.865],  # S1
        [54.970, 36.246, -3.077],  # E2
    ]
)
