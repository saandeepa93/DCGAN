from sys import exit as e
from skimage import io
import click


import modules.util as util
from modules.train import train_gan

configs = util.get_config()

@click.command()
def train():
  train_gan(configs)


@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(train)
  # main.add_command(test)
  main()