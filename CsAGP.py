import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import os, time, glob, random
import yaml
from tqdm import tqdm
import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
pool_rate = 0.5
batch_size = 128

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import torch_geometric.nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv
from torch_geometric.nn import global_mean_pool




def getGraph_8(n=None, batchs=None, class_token=False):

    result = []  # 边的集合
    u = (torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7,
         7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 14,
         14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17,
         18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21,
         21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
         25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29,
         29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32,
         32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36,
         36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39,
         39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43,
         44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47,
         47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50,
         51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54,
         54, 54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58,
         58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62,
         62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65,
         65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69,
         69, 69, 69, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73,
         73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76,
         77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80,
         80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 84, 84, 84,
         84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 87, 87, 88, 88,
         88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91,
         91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 94, 94, 95, 95,
         95, 95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 99, 99, 99, 99,
         99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102,
         102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104, 105,
         105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107,
         107, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110,
         110, 110, 110, 111, 111, 111, 111, 111, 112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 114,
         114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116,
         116, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119,
         119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122,
         122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 124, 125,
         125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128,
         128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130, 130, 131, 131, 131,
         131, 131, 131, 131, 131, 132, 132, 132, 132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133, 133, 134,
         134, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 135, 135, 136, 136, 136, 136, 136, 136, 136,
         136, 137, 137, 137, 137, 137, 137, 137, 137, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139,
         140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 142, 142, 143,
         143, 143, 143, 143, 143, 143, 143, 144, 144, 144, 144, 144, 144, 144, 144, 145, 145, 145, 145, 145, 145, 145,
         145, 146, 146, 146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148, 148,
         148, 148, 148, 149, 149, 149, 149, 149, 149, 149, 149, 150, 150, 150, 150, 150, 150, 150, 150, 151, 151, 151,
         151, 151, 151, 151, 151, 152, 152, 152, 152, 152, 152, 152, 152, 153, 153, 153, 153, 153, 154, 154, 154, 154,
         154, 155, 155, 155, 155, 155, 155, 155, 155, 156, 156, 156, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157,
         157, 157, 157, 158, 158, 158, 158, 158, 158, 158, 158, 159, 159, 159, 159, 159, 159, 159, 159, 160, 160, 160,
         160, 160, 160, 160, 160, 161, 161, 161, 161, 161, 161, 161, 161, 162, 162, 162, 162, 162, 162, 162, 162, 163,
         163, 163, 163, 163, 163, 163, 163, 164, 164, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165, 165, 165,
         165, 166, 166, 166, 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 168, 168, 168, 168, 168, 169, 169, 169,
         169, 169, 169, 169, 169, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 171, 172,
         172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 174, 174, 174, 174, 174, 174, 174,
         174, 175, 175, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177, 177,
         177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179, 179, 179, 179, 179, 179, 179, 180, 180, 180,
         180, 180, 180, 180, 180, 181, 181, 181, 181, 181, 182, 182, 182, 183, 183, 183, 183, 183, 184, 184, 184, 184,
         184, 185, 185, 185, 185, 185, 186, 186, 186, 186, 186, 187, 187, 187, 187, 187, 188, 188, 188, 188, 188, 189,
         189, 189, 189, 189, 190, 190, 190, 190, 190, 191, 191, 191, 191, 191, 192, 192, 192, 192, 192, 193, 193, 193,
         193, 193, 194, 194, 194, 194, 194, 195, 195, 195]))
    v = (torch.tensor(
        [15, 1, 14, 14, 16, 0, 2, 15, 15, 17, 1, 3, 16, 16, 18, 2, 4, 17, 17, 19, 3, 5, 18, 18, 20, 4, 6, 19, 19, 21, 5,
         7, 20, 20, 22, 6, 8, 21, 21, 23, 7, 9, 22, 22, 24, 8, 10, 23, 23, 25, 9, 11, 24, 24, 26, 10, 12, 25, 25, 27,
         11, 13, 26, 26, 12, 27, 1, 29, 0, 15, 28, 0, 2, 28, 30, 1, 14, 16, 29, 1, 3, 29, 31, 2, 15, 17, 30, 2, 4, 30,
         32, 3, 16, 18, 31, 3, 5, 31, 33, 4, 17, 19, 32, 4, 6, 32, 34, 5, 18, 20, 33, 5, 7, 33, 35, 6, 19, 21, 34, 6, 8,
         34, 36, 7, 20, 22, 35, 7, 9, 35, 37, 8, 21, 23, 36, 8, 10, 36, 38, 9, 22, 24, 37, 9, 11, 37, 39, 10, 23, 25,
         38, 10, 12, 38, 40, 11, 24, 26, 39, 11, 13, 39, 41, 12, 25, 27, 40, 12, 40, 13, 26, 41, 15, 43, 14, 29, 42, 14,
         16, 42, 44, 15, 28, 30, 43, 15, 17, 43, 45, 16, 29, 31, 44, 16, 18, 44, 46, 17, 30, 32, 45, 17, 19, 45, 47, 18,
         31, 33, 46, 18, 20, 46, 48, 19, 32, 34, 47, 19, 21, 47, 49, 20, 33, 35, 48, 20, 22, 48, 50, 21, 34, 36, 49, 21,
         23, 49, 51, 22, 35, 37, 50, 22, 24, 50, 52, 23, 36, 38, 51, 23, 25, 51, 53, 24, 37, 39, 52, 24, 26, 52, 54, 25,
         38, 40, 53, 25, 27, 53, 55, 26, 39, 41, 54, 26, 54, 27, 40, 55, 29, 57, 28, 43, 56, 28, 30, 56, 58, 29, 42, 44,
         57, 29, 31, 57, 59, 30, 43, 45, 58, 30, 32, 58, 60, 31, 44, 46, 59, 31, 33, 59, 61, 32, 45, 47, 60, 32, 34, 60,
         62, 33, 46, 48, 61, 33, 35, 61, 63, 34, 47, 49, 62, 34, 36, 62, 64, 35, 48, 50, 63, 35, 37, 63, 65, 36, 49, 51,
         64, 36, 38, 64, 66, 37, 50, 52, 65, 37, 39, 65, 67, 38, 51, 53, 66, 38, 40, 66, 68, 39, 52, 54, 67, 39, 41, 67,
         69, 40, 53, 55, 68, 40, 68, 41, 54, 69, 43, 71, 42, 57, 70, 42, 44, 70, 72, 43, 56, 58, 71, 43, 45, 71, 73, 44,
         57, 59, 72, 44, 46, 72, 74, 45, 58, 60, 73, 45, 47, 73, 75, 46, 59, 61, 74, 46, 48, 74, 76, 47, 60, 62, 75, 47,
         49, 75, 77, 48, 61, 63, 76, 48, 50, 76, 78, 49, 62, 64, 77, 49, 51, 77, 79, 50, 63, 65, 78, 50, 52, 78, 80, 51,
         64, 66, 79, 51, 53, 79, 81, 52, 65, 67, 80, 52, 54, 80, 82, 53, 66, 68, 81, 53, 55, 81, 83, 54, 67, 69, 82, 54,
         82, 55, 68, 83, 57, 85, 56, 71, 84, 56, 58, 84, 86, 57, 70, 72, 85, 57, 59, 85, 87, 58, 71, 73, 86, 58, 60, 86,
         88, 59, 72, 74, 87, 59, 61, 87, 89, 60, 73, 75, 88, 60, 62, 88, 90, 61, 74, 76, 89, 61, 63, 89, 91, 62, 75, 77,
         90, 62, 64, 90, 92, 63, 76, 78, 91, 63, 65, 91, 93, 64, 77, 79, 92, 64, 66, 92, 94, 65, 78, 80, 93, 65, 67, 93,
         95, 66, 79, 81, 94, 66, 68, 94, 96, 67, 80, 82, 95, 67, 69, 95, 97, 68, 81, 83, 96, 68, 96, 69, 82, 97, 71, 99,
         70, 85, 98, 70, 72, 98, 100, 71, 84, 86, 99, 71, 73, 99, 101, 72, 85, 87, 100, 72, 74, 100, 102, 73, 86, 88,
         101, 73, 75, 101, 103, 74, 87, 89, 102, 74, 76, 102, 104, 75, 88, 90, 103, 75, 77, 103, 105, 76, 89, 91, 104,
         76, 78, 104, 106, 77, 90, 92, 105, 77, 79, 105, 107, 78, 91, 93, 106, 78, 80, 106, 108, 79, 92, 94, 107, 79,
         81, 107, 109, 80, 93, 95, 108, 80, 82, 108, 110, 81, 94, 96, 109, 81, 83, 109, 111, 82, 95, 97, 110, 82, 110,
         83, 96, 111, 85, 113, 84, 99, 112, 84, 86, 112, 114, 85, 98, 100, 113, 85, 87, 113, 115, 86, 99, 101, 114, 86,
         88, 114, 116, 87, 100, 102, 115, 87, 89, 115, 117, 88, 101, 103, 116, 88, 90, 116, 118, 89, 102, 104, 117, 89,
         91, 117, 119, 90, 103, 105, 118, 90, 92, 118, 120, 91, 104, 106, 119, 91, 93, 119, 121, 92, 105, 107, 120, 92,
         94, 120, 122, 93, 106, 108, 121, 93, 95, 121, 123, 94, 107, 109, 122, 94, 96, 122, 124, 95, 108, 110, 123, 95,
         97, 123, 125, 96, 109, 111, 124, 96, 124, 97, 110, 125, 99, 127, 98, 113, 126, 98, 100, 126, 128, 99, 112, 114,
         127, 99, 101, 127, 129, 100, 113, 115, 128, 100, 102, 128, 130, 101, 114, 116, 129, 101, 103, 129, 131, 102,
         115, 117, 130, 102, 104, 130, 132, 103, 116, 118, 131, 103, 105, 131, 133, 104, 117, 119, 132, 104, 106, 132,
         134, 105, 118, 120, 133, 105, 107, 133, 135, 106, 119, 121, 134, 106, 108, 134, 136, 107, 120, 122, 135, 107,
         109, 135, 137, 108, 121, 123, 136, 108, 110, 136, 138, 109, 122, 124, 137, 109, 111, 137, 139, 110, 123, 125,
         138, 110, 138, 111, 124, 139, 113, 141, 112, 127, 140, 112, 114, 140, 142, 113, 126, 128, 141, 113, 115, 141,
         143, 114, 127, 129, 142, 114, 116, 142, 144, 115, 128, 130, 143, 115, 117, 143, 145, 116, 129, 131, 144, 116,
         118, 144, 146, 117, 130, 132, 145, 117, 119, 145, 147, 118, 131, 133, 146, 118, 120, 146, 148, 119, 132, 134,
         147, 119, 121, 147, 149, 120, 133, 135, 148, 120, 122, 148, 150, 121, 134, 136, 149, 121, 123, 149, 151, 122,
         135, 137, 150, 122, 124, 150, 152, 123, 136, 138, 151, 123, 125, 151, 153, 124, 137, 139, 152, 124, 152, 125,
         138, 153, 127, 155, 126, 141, 154, 126, 128, 154, 156, 127, 140, 142, 155, 127, 129, 155, 157, 128, 141, 143,
         156, 128, 130, 156, 158, 129, 142, 144, 157, 129, 131, 157, 159, 130, 143, 145, 158, 130, 132, 158, 160, 131,
         144, 146, 159, 131, 133, 159, 161, 132, 145, 147, 160, 132, 134, 160, 162, 133, 146, 148, 161, 133, 135, 161,
         163, 134, 147, 149, 162, 134, 136, 162, 164, 135, 148, 150, 163, 135, 137, 163, 165, 136, 149, 151, 164, 136,
         138, 164, 166, 137, 150, 152, 165, 137, 139, 165, 167, 138, 151, 153, 166, 138, 166, 139, 152, 167, 141, 169,
         140, 155, 168, 140, 142, 168, 170, 141, 154, 156, 169, 141, 143, 169, 171, 142, 155, 157, 170, 142, 144, 170,
         172, 143, 156, 158, 171, 143, 145, 171, 173, 144, 157, 159, 172, 144, 146, 172, 174, 145, 158, 160, 173, 145,
         147, 173, 175, 146, 159, 161, 174, 146, 148, 174, 176, 147, 160, 162, 175, 147, 149, 175, 177, 148, 161, 163,
         176, 148, 150, 176, 178, 149, 162, 164, 177, 149, 151, 177, 179, 150, 163, 165, 178, 150, 152, 178, 180, 151,
         164, 166, 179, 151, 153, 179, 181, 152, 165, 167, 180, 152, 180, 153, 166, 181, 155, 183, 154, 169, 182, 154,
         156, 182, 184, 155, 168, 170, 183, 155, 157, 183, 185, 156, 169, 171, 184, 156, 158, 184, 186, 157, 170, 172,
         185, 157, 159, 185, 187, 158, 171, 173, 186, 158, 160, 186, 188, 159, 172, 174, 187, 159, 161, 187, 189, 160,
         173, 175, 188, 160, 162, 188, 190, 161, 174, 176, 189, 161, 163, 189, 191, 162, 175, 177, 190, 162, 164, 190,
         192, 163, 176, 178, 191, 163, 165, 191, 193, 164, 177, 179, 192, 164, 166, 192, 194, 165, 178, 180, 193, 165,
         167, 193, 195, 166, 179, 181, 194, 166, 194, 167, 180, 195, 169, 168, 183, 168, 170, 169, 182, 184, 169, 171,
         170, 183, 185, 170, 172, 171, 184, 186, 171, 173, 172, 185, 187, 172, 174, 173, 186, 188, 173, 175, 174, 187,
         189, 174, 176, 175, 188, 190, 175, 177, 176, 189, 191, 176, 178, 177, 190, 192, 177, 179, 178, 191, 193, 178,
         180, 179, 192, 194, 179, 181, 180, 193, 195, 180, 181, 194]))
    edge_index = torch.stack((u, v), dim=0).long().to(device)
    result = [edge_index] * int(batchs)
    return result


class GSAPool(torch.nn.Module):

    def __init__(self, in_channels, num_nodes, pooling_ratio=0.9, batch_size=2, alpha=0.6, pooling_conv="GCNConv",
                 fusion_conv="GATConv",
                 min_score=None, multiplier=1, device=True, non_linearity=torch.tanh):
        super(GSAPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = pooling_ratio
        self.alpha = alpha

        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer1 = nn.Conv1d(in_channels, 1, 1)
        self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)
        self.combine = nn.Conv1d(num_nodes * batch_size, num_nodes * batch_size, 1)

        self.min_score = min_score
        self.multiplier = multiplier

        self.fusion_flag = 0
        if (fusion_conv != "false"):
            self.fusion_flag = 1
        self.non_linearity = non_linearity
        self.device = device



    def conv_selection(self, conv, in_channels, conv_type=0):
        if (conv_type == 0):
            out_channels = 1
        elif (conv_type == 1):
            out_channels = in_channels
        if (conv == "GCNConv"):
            return GCNConv(in_channels, out_channels)
        elif (conv == "ChebConv"):
            return ChebConv(in_channels, out_channels, 1)
        elif (conv == "SAGEConv"):
            return SAGEConv(in_channels, out_channels)
        elif (conv == "GATConv"):
            return GATConv(in_channels, out_channels, heads=1, concat=True)
        elif (conv == "GraphConv"):
            return GraphConv(in_channels, out_channels)
        else:
            raise ValueError

    def forward_feture(self, x, index):
        Batch_size, num_node, dim = x.size()
        if index == 0:
            graph = getGraph_8(int((num_node - 1) ** 0.5), Batch_size)
        else:
            graph = euclidean_dist(x[:, 1:], x[:, 1:])
        class_tokens = x[:, :1]
        result = []
        for i in range(len(graph)):
            if not self.device:
                result.append(Data(x=x[i][1:], edge_index=graph[i]))
            else:
                result.append(Data(x=x[i][1:], edge_index=graph[i]).to(device))
        result = Batch.from_data_list(result)  # 小图封装成大图
        return result, class_tokens, dim, Batch_size

    def forward(self, x, index, edge_attr=None):

        x, class_tokens, dim, Batch_size = self.forward_feture(x, index)
        x, edge_index, batch = x.x, x.edge_index, x.batch
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        #         print(x.shape)
        # SBTL
        score_s = self.sbtl_layer(x, edge_index).squeeze()
        # FBTL
        score_f = self.fbtl_layer1(x.permute(1, 0)).squeeze()
        # hyperparametr alpha
        score = (score_s + score_f).unsqueeze(-1).float()
        score = self.combine(score).squeeze()
        score = (score_s+score_f)
        score = score.unsqueeze(-1) if score.dim() == 0 else score

        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = F.softmax(score, batch)
        perm = topk(score, self.ratio, batch)

        # fusion
        if (self.fusion_flag == 1):
            x = self.fusion(x, edge_index)

        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        x = x.reshape(Batch_size, -1, dim)
        x = torch.cat((class_tokens, x), dim=1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, pool_rate, batch_size, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:
                self.fusion.append(
                    CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                        has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.gsapool = GSAPool(in_channels=dim[0], num_nodes=patches[0], pooling_ratio=pool_rate, batch_size=batch_size)

    def forward(self, x, index):
        mri, pet = x
        block1, block2 = self.blocks

        mri = block2(mri)
        pet = block1(pet)

        outs_b = [mri, pet]

        proj_cls_token = [outs_b[0][:, 0:1], outs_b[1][:, 0:1]]
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = tmp[:, 0:1, ...]
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)

            tmp, mri_edge_index, edge_attr, batch, perm = self.gsapool(tmp, index)
            outs.append(tmp)

        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size, patches)]


def euclidean_dist(x, y, class_token=False):
    b, m, n = x.size(0), x.size(1), y.size(1)
    edge_indexs = []
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(b, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(b, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist += -2 * (x @ y.permute(0, 2, 1))
    dist = dist.clamp(min=1e-12).sqrt()
    for i in range(b):
        avg = torch.mean(dist[i])
        edges = torch.nonzero(dist[i] <= avg)
        if not class_token:
            edge = torch.stack((edges[:, 0], edges[:, 1]), dim=0)
            edge_indexs.append(edge)

        else:
            u, v = edges[:, 0], edges[:, 1]
            cls1 = torch.tensor([])
            cls2 = torch.tensor([])
            for j in range(m):
                cls1 = torch.cat((cls1, torch.tensor([m])), -1)
                cls2 = torch.cat((cls2, torch.tensor([j])), -1)
            class_token1 = torch.cat((cls1, cls2), -1).int().to(device)
            class_token2 = torch.cat((cls2, cls1), -1).int().to(device)

            u = torch.cat((u, class_token2), -1)
            v = torch.cat((v, class_token1), -1)
            edge = torch.stack((u, v), dim=0).long()
            edge_indexs.append(edge)
    return edge_indexs


class VisionTransformer(nn.Module):


    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, pool_rate=0.9, batch_size=2, num_classes=3,
                 embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()
        self.pool_rate = pool_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        # print(num_patches)
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:

            self.pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(
                    PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))

    
        self.cls_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]

            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, pool_rate=pool_rate, batch_size=batch_size,
                                  num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr_,
                                  norm_layer=norm_layer)
            num_patches = [math.ceil(i * self.pool_rate) for i in num_patches]

            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in
                                   range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, mri, pet):
        B, C, H, W = mri.shape
        imgs = [mri, pet]
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(imgs[i], size=(self.img_size[i], self.img_size[i]),
                                                 mode='bicubic') if H != self.img_size[i] else imgs[i]
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for index, blk in enumerate(self.blocks):
            xs = blk(xs, index)

        for x in xs: x[:, 0] = torch.sum(x, dim=1)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def forward(self, mri, pet):
        xs = self.forward_features(mri, pet)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        return ce_logits


def CsAGP(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[224, 224],
                              patch_size=[16, 16], embed_dim=[256, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, pool_rate=pool_rate,
                              batch_size=batch_size, **kwargs)
    return model





