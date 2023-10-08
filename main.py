import os
import argparse as ap

from questions import q1, q2, q3


def main(args):

    if args.question == 'q1':
        q1(args)

    if args.question == 'q2':
        q2(args)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3', 'q4', 'q5'], required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--new_annot', action='store_true')
    parser.add_argument('-i', '--image', default=None)
    parser.add_argument('-v', '--viz', default='lines')
    args = parser.parse_args()
    main(args)
