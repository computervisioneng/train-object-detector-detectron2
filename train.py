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
    ----- val
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
    parser.add_argument('--class-list', default='./class.names')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learning-rate', default=0.00025)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--iterations', default=10000)
    parser.add_argument('--checkpoint-period', default=500)
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

    args = parser.parse_args()

    util.train(args.output_dir,
               args.data_dir,
               args.class_list,
               device=args.device,
               learning_rate=float(args.learning_rate),
               batch_size=int(args.batch_size),
               iterations=int(args.iterations),
               checkpoint_period=int(args.checkpoint_period),
               model=args.model)
