import os
import argparse

import util


if __name__ == "__main__":
    """
    annotations should be provided in yolo format, this is: 
            class, xc, yc, w, h
    data needs to follow this structure:
    
    data-dir
    ----- train
    --------- imgs
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='.')
    parser.add_argument('--class-list', default='./classes.names')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    root_dir = args.root_dir

    class_list_file = args.class_list
    data_dir = args.data_dir
    output_dir = args.output_dir
    device = args.device

    util.train(output_dir, data_dir, class_list_file, device=device)
