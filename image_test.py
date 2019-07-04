#-*- coding:utf-8 -*-
from utils.point_feature import *


def main():
    print("Test for the image.")
    load_frozen_model('data/geodesc.pb', print_nodes=True)


if __name__ == '__main__':
    main()
