
import models
import pandas as pd
import os 
import models
from sklearn.preprocessing import StandardScaler
import datetime
import argparse
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from runner.Runner import Runner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_path = "data/tep_sample/tep_13_y_15.csv"
df = pd.read_csv(data_path, index_col=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomlay detection at Tenesee Eastoman Process example',
                        help='trainning data to learn')
    parser.add_argument('--outputdir', '-o', type=str, default=None,
                        help='output directory name')
    parser.add_argument('--optimize', '-op', type=bool, default=True,
                        help='optimise_parmas')
    parser.add_argument('--closs_valid', '-cv', type=int, default=5,
                        help='num closs validation if 0 not cv ')
    args = parser.parse_args()
    if args.outputdir is None:
        fname, ext = os.path.splitext(os.path.basename(__file__))
        today = datetime.datetime.today()
        outputdir = os.path.join("log",
                                 today.strftime("%Y%m%d_%H%M%S_" + fname))
    else:
        outputdir = args.outputdir
        if not outputdir.startswith("log/"):
            outputdir = os.path.join("log", outputdir)
            os.makedirs(outputdir, exist_ok=True)
    exog_columns = df.columns.tolist()[1:]
    df_x = df[exog_columns]
    endog_column = ["y"]
    df_y = df[endog_column]
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.666, shuffle =False)
    train_models = {
                    "IF": models.IFModel(base_dir=outputdir),
                    "PCA": models.PCAModel(base_dir=outputdir),
                    "OCSVM": models.OCSVMModel(base_dir=outputdir),
                    "LOF": models.LOFModel(base_dir=outputdir),
                    "AE": models.AEModel(base_dir=outputdir)
                    }
    trained_models = {}
    for modelname in train_models:
        model = train_models[modelname]
        model.fit(x_train)
        pred = model.decision_function(x_test)
        plt.figure()
        plt.plot(np.arange(len(pred)), pred)
        plt.title("Anomaly detection Model:{}".format(modelname))
        trained_models[modelname] = model