from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, shutil

import _init_paths
from control.train import Train
from control.evaluate import Evaluate
from control.prepare_dataset import PrepareDataset
from utils.config import global_config

import colored_traceback.always

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--mode', help='mode should be one of "train" "new_train" "eval" "inference"', required=True)
parser.add_argument('-d','--device', required=False)
args = vars(parser.parse_args())

def main():
    global_config.assign_config()

    if args['device'] == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rm_file("./data/log/*log.txt")

    if args['mode'] == "train":
        Train().run()
    elif args['mode'] == "new_train":
        print("Clean chceck point: train_dir ")
        clean_folder(global_config.global_config.train_dir)
        print("Clean log directory")
        clean_folder(global_config.global_config.log_dir)
        Train().run()
    elif args['mode'] == "eval":
        Evaluate().run()

# def prepare_dataset():
#     global_config.assign_config()
#     preparedb=PrepareDataset()
#     imdb, roidb = preparedb.combined_roidb('voc_2007_trainval')
#     print('{:d} roidb entries'.format(len(roidb)))

def clean_folder(folder_dir):
    for the_file in os.listdir(folder_dir):
        file_path = os.path.join(folder_dir, the_file)
        rm_file(file_path)

def rm_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
    # prepare_dataset()
