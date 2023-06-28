import sys
from wgan_gp import WGAN_GP

if __name__ == "__main__":
    model=WGAN_GP()
    model.train()
