from ast import literal_eval
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_summaries = [
    '../ACE/ACE/results_summaries/mixed8_cab_cropped_n_clusters_15_ace_results.txt',
    '../ACE/ACE/results_summaries/mixed8_cab_cropped_n_clusters_20_ace_results.txt',
    '../ACE/ACE/results_summaries/mixed8_cab_cropped_n_clusters_25_ace_results.txt'
]

all_tcav_scores = []

for results in results_summaries:
    
    n_clusters = '_'.join(results.split('/')[-1].split('_')[3:6])
    filename = results.split('/')[-1].split('_')

    if len(filename) < 6:
        layer, label = filename[:2]
    else:
        layer = filename[0]
        label = '_'.join(filename[1:4])

    with open(results) as f:
        lines = f.readlines()
        lines.append('END OF FILE')

    try:
        block_1 = lines.index('\t\t\t ---CAV accuracies---\n')
        block_2 = lines.index('\t\t\t ---Raw CAV accuracies data---\n')
        block_3 = lines.index('\t\t\t ---TCAV scores---\n')
        block_4 = lines.index('\t\t\t ---Raw TCAV scores data---\n')
        block_5 = lines.index('END OF FILE')
    except ValueError:
        continue

    summary_tcav_scores = lines[block_3+2:block_4]
    all_raw_tcav_scores = lines[block_4+2:block_5]

    tcav_concept_num = []
    tcav_score_mean = []
    tcav_score_std = []
    tcav_non_sig_concepts = []

    for idx, tcav_score in enumerate(summary_tcav_scores):

        tcav_score = tcav_score.strip()
        if len(tcav_score.split(':')) > 3:
            if tcav_score.split(':')[2] == 'overall':
                layer, label_concept, _, scores = tcav_score.split(':')
            else:
                continue
        else: 
            layer, label_concept, scores = tcav_score.split(':')

        try:           
            concept = label_concept.split('_')[-1]
            pval = float(scores.split(',')[1])
            mean, std = scores.split(',')[0].split('±')
            mean = float(mean)
            std = float(std)

            # Toggle Bonferroni correction for combined concepts
            # if pval >= (0.01 / 3):
            if pval >= 0.01:
                tcav_non_sig_concepts.append(concept)

            _, concept_num = concept.split('concept')
            tcav_concept_num.append(concept_num)
            tcav_score_mean.append(mean)
            tcav_score_std.append(std)

            # results_summaries_dict[label][layer][concept]['tcav_mean'] = mean
            # results_summaries_dict[label][layer][concept]['tcav_std'] = std
            # results_summaries_dict[label][layer][concept]['tcav_pval'] = pval
        except NameError:
            pass
    
    n_cluster_tcav_scores = []
    for raw_score in all_raw_tcav_scores:
        
        raw_score = raw_score.strip()
        if len(raw_score.split(':')) > 3:
            if raw_score.split(':')[2] == 'overall':
                layer, label_concept, _, scores = raw_score.split(':')
            else:
                continue
        else: 
            layer, label_concept, scores = raw_score.split(':')

        try:
            concept = label_concept.split('_')[-1]
            scores = literal_eval(scores)

            if concept not in tcav_non_sig_concepts:
                n_cluster_tcav_scores.extend(scores)

            # results_summaries_dict[label][layer][concept]['all_tcav_scores'] = scores
        except NameError:
            pass
    
    all_tcav_scores.append(n_cluster_tcav_scores)

plt.boxplot(all_tcav_scores)
plt.title('Overall TCAV Scores by Cluster Number')
plt.xticks([1, 2, 3], ['15', '20', '25'])
plt.xlabel('No. Clusters')
plt.ylabel('TCAV Scores')
plt.savefig(f'./tcav_scores_plots/boxplot_{label}_{layer}_{n_clusters}_TCAV_scores.png')
plt.clf()
plt.cla()
plt.close()
