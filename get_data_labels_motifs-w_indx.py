#InteractionsFileCaptureC='/srv/scratch/mtaranov/CaptureC_ChicagoCalls/CaptureC_SC_bait-bait.bed.gz'
#run on kali
InteractionsFileCaptureC='/users/mtaranov/CaptureC_files/CaptureC_SC_bait-bait.bed.gz'
#PromoterFile='/srv/scratch/mtaranov/HindIII/PromoterCapture_Digest_Human_HindIII_baits_ID.bed'
#run on kali
PromoterFile='/users/mtaranov/CaptureC_files/PromoterCapture_Digest_Human_HindIII_baits_ID.bed'
motifs_top3='/users/mtaranov/sequence_to_motif/out_3_0.0001/mat.npy'
motifs_top3_pos='/users/mtaranov/sequence_to_motif/out_3_0.0001/pos.npy'
PROJDIR='/users/mtaranov/NN_thres5max_datasets/'
DATAJDIR='/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/'
#features_path='/srv/scratch/mtaranov/peaks_at_promoters/output_thres5_max/'
#run on kali
features_path='/users/mtaranov/CaptureC_files/output_thres5_max/'
atac=features_path+'atac'
#TFs
ctcf=features_path+'CTCF'
pol3=features_path+'PolII'
TP63=features_path+'TP63'
LSD1=features_path+'LSD1'
#histone
H3K27ac=features_path+'H3K27ac'
H3K27me3=features_path+'H3K27me3'
H3K4me1=features_path+'H3K4me1'
H3K4me3=features_path+'H3K4me3'
H3K9ac=features_path+'H3K9ac'


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
import math
import scipy.linalg
import itertools
import copy
import random
import gzip
from sets import Set
#reload(rcca)
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from utils_data_process import printMatrix, set_diag_to_value, binarize, binarize_w_unlabeled, demean, zscore
from utils_data_process import shuffle_nodes, train_vali_test, build_distance_for_node, BuildMatrix, get_features
from utils_data_process import get_data_labels, remove_unlabeled, concatenate_chrs, get_pairs_distance_matched, impose_dist_constrains, count_nodes_and_contacts

labels_score = BuildMatrix(PromoterFile, InteractionsFileCaptureC)

VectorATAC=get_features(PromoterFile, atac, 'atac')

VectorCTCF=get_features(PromoterFile, ctcf, 'ctcf')
VectorPOL3=get_features(PromoterFile, pol3, 'pol3')
VectorTP63=get_features(PromoterFile, TP63, 'TP63')
VectorLSD1=get_features(PromoterFile, LSD1, 'LSD1')

VectorH3K27ac=get_features(PromoterFile, H3K27ac, 'H3K27ac')
VectorH3K27me3=get_features(PromoterFile, H3K27me3, 'H3K27me3')
VectorH3K4me1=get_features(PromoterFile, H3K4me1, 'H3K4me1')
VectorH3K4me3=get_features(PromoterFile, H3K4me3, 'H3K4me3')
VectorH3K9ac=get_features(PromoterFile, H3K9ac, 'H3K9ac')


distance_for_node = build_distance_for_node(PromoterFile)

FeatureVector_wo_dist = {}
FeatureVector = {}
for chr in VectorATAC:
    if chr != 'chrY':
        FeatureVector_wo_dist[chr] = demean(np.vstack((VectorATAC[chr], VectorCTCF[chr], VectorPOL3[chr], VectorTP63[chr], VectorLSD1[chr], VectorH3K27ac[chr], VectorH3K27me3[chr], VectorH3K4me1[chr], VectorH3K4me3[chr], VectorH3K9ac[chr])).T)
        FeatureVector[chr] = np.concatenate((FeatureVector_wo_dist[chr], distance_for_node[chr]), axis=1)


pos=np.load(motifs_top3_pos)
mat=np.load(motifs_top3)
all_chr=['chr'+str(i+1) for i in range(22)]+['chrX']

FeatureAndMotifVector={}
VectorMotif={}
for chr in all_chr:
    indx = np.where(pos[:,0]==chr)
    VectorMotif[chr]=mat[indx]
    FeatureAndMotifVector[chr]=np.concatenate((FeatureVector[chr], VectorMotif[chr]), axis=1)

data, labels, indx  = get_data_labels(FeatureAndMotifVector, labels_score)
#data, labels, indx  = get_data_labels(FeatureVector, labels_score)
data_all_chrs, labels_all_chrs, indx_all_chr = concatenate_chrs(data, labels, indx)
data_all_wo_unlbd_at_10, labels_all_wo_unlbd_at_10, indx_all_wo_unlbd_at_10 = remove_unlabeled(data_all_chrs, binarize_w_unlabeled(labels_all_chrs, 10), indx_all_chr)
train_set_thres_10, vali_set_thres_10, test_set_thres_10, labels_train_thres_10, labels_vali_thres_10, labels_test_thres_10, indx_train_thres_10, indx_vali_thres_10, indx_test_thres_10 =train_vali_test(data_all_wo_unlbd_at_10, labels_all_wo_unlbd_at_10, indx_all_wo_unlbd_at_10, 0.5, 0.3)

min_dist=10000
max_dist=2000000
dist_step=10000
class_imbalance=1
X_test_distance_matched_at_10, y_test_distance_matched_at_10, indx_test_distance_matched_at_10 = get_pairs_distance_matched(test_set_thres_10, labels_test_thres_10, indx_test_thres_10, min_dist, max_dist, dist_step, class_imbalance)
X_train_distance_matched_at_10, y_train_distance_matched_at_10, indx_train_distance_matched_at_10 = get_pairs_distance_matched(train_set_thres_10, labels_train_thres_10, indx_train_thres_10, min_dist, max_dist, dist_step, class_imbalance)
X_valid_distance_matched_at_10, y_valid_distance_matched_at_10, indx_valid_distance_matched_at_10 = get_pairs_distance_matched(vali_set_thres_10, labels_vali_thres_10, indx_vali_thres_10, min_dist, max_dist, dist_step, class_imbalance)


print "in all chrs:"
y_at_10=np.concatenate((y_train_distance_matched_at_10, y_valid_distance_matched_at_10, y_test_distance_matched_at_10), axis=0)
print "pos at thres=10: ", np.where(y_at_10 > 0)[0].shape[0], " | train: ", np.where(y_train_distance_matched_at_10 > 0)[0].shape[0], "vali:",np.where(y_valid_distance_matched_at_10 > 0)[0].shape[0], "test:",np.where(y_test_distance_matched_at_10 > 0)[0].shape[0] 
print "neg at thres=10: ", np.where(y_at_10 == 0)[0].shape[0], " | train: ", np.where(y_train_distance_matched_at_10 == 0)[0].shape[0], "vali:",np.where(y_valid_distance_matched_at_10 == 0)[0].shape[0], "test:",np.where(y_test_distance_matched_at_10 == 0)[0].shape[0]
print "total: ", y_at_10.shape[0]

count_nodes_and_contacts(indx_train_distance_matched_at_10, indx_valid_distance_matched_at_10, indx_test_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/X_train_thres_10_w_motifs.npy', X_train_distance_matched_at_10) 
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/X_valid_thres_10_w_motifs.npy', X_valid_distance_matched_at_10) 
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/X_test_thres_10_w_motifs.npy', X_test_distance_matched_at_10) 
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/y_train_thres_10_w_motifs.npy', y_train_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/y_valid_thres_10_w_motifs.npy', y_valid_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/y_test_thres_10_w_motifs.npy', y_test_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/indx_train_thres_10_w_motifs.npy', indx_train_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/indx_valid_thres_10_w_motifs.npy', indx_valid_distance_matched_at_10)
np.save('/users/mtaranov/NN_thres5max_datasets/dist_matched/w_motifs/indx_test_thres_10_w_motifs.npy', indx_test_distance_matched_at_10)
