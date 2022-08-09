from cmath import sqrt
from mimetypes import init
import os
import random
import argparse
from turtle import distance
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg
import numpy as np
from torchvision.utils import save_image
from utils.weights import load_from_weights

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--data_folder', type=str, help="folder containing the data (crops)")
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args

def get_value_from_key(d, val):
    for key, v in d.items():
        if key == val:
            return v

def main(args):

    # model
    model = ResNet18Backbone(pretrained=False)
    model = load_from_weights(model, args.weights_init, logger=None)
    #raise NotImplementedError("TODO: build model and load weights snapshot")
    

    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = DataReaderPlainImg(os.path.join(args.data_folder, "images/256/val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True)
    #print(val_loader.size())
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [25]
    nns = []
    for idx, img in enumerate(val_loader):

        if idx not in query_indices:
            continue
        query_idx = idx
        print("Computing NNs for sample {}".format(idx))
        # K = 6 because one image will be the query image itself in the results
        closest_idx, closest_dist = find_nn(model, img, val_loader, 6)

    for idx, img in enumerate(val_loader):
        if idx in closest_idx:
            nns.append(img)
            save_image(img, 'result_img/img'+'_query_idx_'+str(query_idx)+'_img_idx_'+str(idx)+'.jpg')

        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    with torch.no_grad():
        save_image(query_img, 'result_img/query_img.jpg')
        query_img_output = model(query_img.cuda())
        l2_dist = dict()
        for id, img_temp in enumerate(loader):
                loader_img_output = model(img_temp.cuda())

                # Calculating nearest neighbors and updating values to dictionary
                difference = query_img_output - loader_img_output
                l2_value = difference * difference
                l2_value = torch.sum(l2_value)
                l2_value = torch.sqrt(l2_value.detach().cuda())
                l2_dist.update({id:l2_value})
    
    # sorting to get key with lowest values
    closest_idx = [y[0] for y in sorted(l2_dist.items(), key = lambda x: x[1])[:k]]
    closest_dist = []

    # obtaining the closest values
    for value in closest_idx:
        val_temp = get_value_from_key(l2_dist, value)
        closest_dist.append(val_temp)



    #raise NotImplementedError("TODO: nearest neighbors retrieval")
    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    #pprint(vars(args))
    #print()
    main(args) 
