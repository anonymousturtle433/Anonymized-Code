import argparse
import sys
from NL_constraints_data_prep.Utils.Dataset import setup_datasets, prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasets", type=str, help="The names of the datasets to use", default="synthetic")

args = parser.parse_args()

if __name__ == "__main__":
  prepare_data(args.datasets)
  print('Jsons generated successfully')