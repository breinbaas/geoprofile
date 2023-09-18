# example
# run file as
# /bin/python /home/breinbaas/dev/profileGAN/src/cpt2csv.py -p data/ -t 0.0 -b -13.0
# data path = ./data
# top = 0.0
# bottom = -13.0

import getopt
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from leveelogic.objects.cpt import Cpt

IMG_WIDTH = 512
IMG_HEIGHT = 32

# this should be automated later
# now extracted from the geotechnical profile
X_MIN = 600
X_MAX = 1000
DIST_LIST = {1441: 615, 1443: 700, 1445: 790, 1444: 880, 1107: 995}


def main():
    options, _ = getopt.getopt(sys.argv[1:], "p:t:b:", ["path =", "top =", "bottom ="])

    cpt_path = ""
    top = 0.0
    bottom = 0.0

    for o, a in options:
        if o in ("-p", "--path"):
            cpt_path = a
        elif o in ("-t", "--top"):
            top = float(a)
        elif o in ("-b", "--bottom"):
            bottom = float(a)

    # defaults for testing
    if cpt_path == "":
        cpt_path = "./data/cpts"
    if top == 0.0:
        top = 0.0
    if bottom == 0.0:
        bottom = -13.0

    # create the input image
    M = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    for cpt_file in [Path(cpt_path, c) for c in os.listdir(cpt_path)]:
        cpt = Cpt.from_file(cpt_file)
        cpt_num = int(cpt_file.stem.split("-")[1])

        a = cpt.to_depth_Ic_list()  # z, Ic

        # plot the original CPT
        plt.figure(figsize=(6, 10))
        plt.plot(cpt.qc, cpt.z, label="qc")
        plt.plot(cpt.fr, cpt.z, label="fr")
        plt.plot(a[:, 1], cpt.z, label="Ic")
        plt.legend()
        plt.savefig(f"./output/{cpt.name}.png")
        plt.clf()

        # apply limits
        a = a[(a[:, 0] >= bottom) & (a[:, 0] <= top)]

        # resize to 32px
        min_y_new = 0
        max_y_new = IMG_HEIGHT - 1

        df = pd.DataFrame({"z": a[:, 0], "Ic": a[:, 1]})
        df["z"] = (top - df["z"]) / (top - bottom) * (max_y_new - min_y_new)
        df["z"] = np.round(df["z"])

        # add the x value
        # df["x"] = np.zeros(df.shape[0])
        # df["x"] = round((DIST_LIST[cpt_num] - X_MIN) / (X_MAX - X_MIN) * IMG_WIDTH)

        x = int(round((DIST_LIST[cpt_num] - X_MIN) / (X_MAX - X_MIN) * IMG_WIDTH))
        print(x)

        # aggregate data, choose one
        # MEAN
        df_new = df.groupby("z").mean().reset_index()
        # GEOMEAN
        # df_new = df.groupby('depth').apply(lambda x: np.exp(np.mean(np.log(x.iloc[:, 1:]), axis=0))).reset_index()
        # MAX
        # df_new = df.groupby('depth').max().reset_index()

        # update the matrix
        M[:, x] = df_new["Ic"].values

    plt.figure(figsize=(10, 3))
    plt.imshow(M)
    plt.colorbar(orientation="horizontal")
    plt.savefig(f"./output/model_input.png")

    df = pd.DataFrame(M)
    df.to_csv("./output/df.csv")

    path_to_model_to_evaluate = r"D:\Documents\Development\Github\schemaGAN\output"
    generator = os.path.join(path_to_model_to_evaluate, "final_generator.h5")
    model = load_model(generator, compile=False)

    # normalize Ic

    # cs_to_evaluate = cs_to_evaluate.reshape(1, 32, 512, 1)
    # normalized_data = IC_normalization(data_to_norm)

    # use model for prediction
    gan_res = model.predict(M)

    # gan_res = reverse_IC_normalization(gan_res)
    # gan_res = np.squeeze(gan_res)
    # plt.imshow(gan_res)
    # plt.colorbar(orientation="horizontal")
    # plt.savefig(f"./output/model_output.png")


if __name__ == "__main__":
    main()
