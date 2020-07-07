from sys import exit as e
from skimage import io


import modules.util as util
from modules.train.tmain import train


def main():
  configs = util.get_config()
  train(configs)



if __name__ == '__main__':
  main()