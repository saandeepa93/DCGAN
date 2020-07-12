from sys import exit as e
from skimage import io
import click
from sys import exit as e


import modules.util as util
from modules.train import train_gan


@click.command()
@click.option('--config', help='path of config file')
def train(config):
  dataset = config.split('/')[-1].split('.')[0]
  configs = util.get_config(config)
  train_gan(configs, dataset)


@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(train)
  # main.add_command(test)
  main()