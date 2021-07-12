from ast import literal_eval
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_summaries = glob('../ACE/ACE/results_summaries/*.txt')

results_summaries = [
    # '../ACE/ACE/results_summaries/mixed8_ambulance_and_cab_ace_results.txt',
    # '../ACE/ACE/results_summaries/mixed8_cab_and_bullet_train_ace_results.txt',
    # '../ACE/ACE/results_summaries/mixed8_cab_and_jeep_ace_results.txt',
    # '../ACE/ACE/results_summaries/mixed8_cab_and_mantis_ace_results.txt'
    '../ACE/ACE/results_summaries/mixed8_cab_ace_results.txt'
]

for results in results_summaries:
    print(results)

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

    summary_cav_accuracies = lines[block_1+2:block_2]
    all_raw_cav_accuracies = lines[block_2+2:block_3-1]
    summary_tcav_scores = lines[block_3+2:block_4]
    all_raw_tcav_scores = lines[block_4+2:block_5]

    # results_summaries_dict = {}
    # results_summaries_dict[label] = {}
    # results_summaries_dict[label][layer] = {}

    cav_concept_num = []
    cav_acc_mean = []
    cav_acc_std = []

    for acc in summary_cav_accuracies:

        acc = acc.strip()
        layer, label_concept, mean_std = acc.split(':')
        concept = label_concept.split('_')[-1]
        mean, std = mean_std.split('±')
        mean = float(mean)
        std = float(std)

        _, concept_num = concept.split('concept')
        cav_concept_num.append(concept_num)
        cav_acc_mean.append(mean)
        cav_acc_std.append(std)

        # results_summaries_dict[label][layer][concept] = {}
        # results_summaries_dict[label][layer][concept]['cav_mean'] = mean
        # results_summaries_dict[label][layer][concept]['cav_std'] = std

    all_cav_concept_num = []
    all_cav_accs = []

    for raw_acc in all_raw_cav_accuracies:

        raw_acc = raw_acc.strip()
        layer, label_concept, accs = raw_acc.split(':')
        concept = label_concept.split('_')[-1]
        accs = literal_eval(accs)

        _, concept_num = concept.split('concept')
        all_cav_concept_num.append(concept_num)
        all_cav_accs.append(accs)

        # results_summaries_dict[label][layer][concept]['all_cav_acc'] = accs

    tcav_concept_num = []
    tcav_score_mean = []
    tcav_score_std = []
    tcav_non_sig = []

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
                tcav_non_sig.append(idx)

            _, concept_num = concept.split('concept')
            tcav_concept_num.append(concept_num)
            tcav_score_mean.append(mean)
            tcav_score_std.append(std)

            # results_summaries_dict[label][layer][concept]['tcav_mean'] = mean
            # results_summaries_dict[label][layer][concept]['tcav_std'] = std
            # results_summaries_dict[label][layer][concept]['tcav_pval'] = pval
        except NameError:
            pass

    all_tcav_concept_num = []
    all_tcav_scores = []

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

            _, concept_num = concept.split('concept')
            all_tcav_concept_num.append(concept_num)
            all_tcav_scores.append(scores)

            # results_summaries_dict[label][layer][concept]['all_tcav_scores'] = scores
        except NameError:
            pass

    plt.scatter(cav_concept_num, cav_acc_mean)
    plt.errorbar(cav_concept_num, cav_acc_mean, yerr=cav_acc_std, fmt="o") 
    # plt.ylim([0.88, 1.02])
    plt.title(f'{label.capitalize()} {layer.capitalize()} CAV Concept Accuracies')
    for idx in tcav_non_sig:
        offset = idx + -0.15
        plt.text(offset, 0.885, '*')
    plt.xlabel('Concept')
    plt.ylabel('CAV Accuracy')
    plt.savefig(f'./cav_accuracies_plots/{label}_{layer}_CAV_concept_accuracy.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.scatter(cav_concept_num, cav_acc_mean)
    plt.errorbar(cav_concept_num, cav_acc_mean, yerr=cav_acc_std, fmt="o") 
    plt.ylim([0.6, 1.03])
    plt.title(f'{label.capitalize()} {layer.capitalize()} CAV Concept Accuracies')
    for idx in tcav_non_sig:
        offset = idx + -0.15
        plt.text(offset, 0.6, '*')
    plt.xlabel('Concept')
    plt.ylabel('CAV Accuracy')
    plt.savefig(f'./cav_accuracies_plots/{label}_{layer}_CAV_concept_accuracy2.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.hist(all_cav_accs[-1])
    plt.title(f'{label.capitalize()} Concept {all_cav_concept_num[-1]} CAV Concept Accuracies')
    plt.xlabel('CAV Accuracy')
    plt.ylabel('Frequency')
    plt.savefig(f'./cav_accuracies_histograms/{label}_{layer}_concept_{all_cav_concept_num[-1]}_CAV_concept_accuracy_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.scatter(tcav_concept_num, tcav_score_mean)
    plt.errorbar(tcav_concept_num, tcav_score_mean, yerr=tcav_score_std, fmt="o") 
    print(np.mean([x for y in all_tcav_scores for x in y]))
    print(np.std([x for y in all_tcav_scores for x in y]))
    plt.axhline(y = np.mean([x for y in all_tcav_scores for x in y]), color = 'r', linestyle = '-')
    plt.ylim([0, 1])
    plt.title(f'{label.capitalize()} {layer.capitalize()} TCAV Scores')
    for idx in tcav_non_sig:
        offset = idx + -0.15
        plt.text(offset, 0, '*')
    plt.xlabel('Concept')
    plt.ylabel('TCAV Score')
    plt.savefig(f'./tcav_scores_plots/{label}_{layer}_TCAV_scores.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.boxplot(all_tcav_scores)
    # plt.axhline(y = np.mean([x for y in all_tcav_scores for x in y]), color = 'r', linestyle = '-')
    plt.title(f'{label.capitalize()} TCAV Scores (Ordered by Decreasing TCAV Score)')
    plt.xlabel('Concept')
    plt.ylabel('TCAV Scores')
    plt.savefig(f'./tcav_scores_plots/boxplot_{label}_{layer}_TCAV_scores.png')
    plt.clf()
    plt.cla()
    plt.close()


    plt.hist(all_tcav_scores[-1])
    plt.title(f'{label.capitalize()} Concept {all_tcav_concept_num[-1]} TCAV Scores')
    plt.xlabel('TCAV Scores')
    plt.ylabel('Frequency')
    plt.savefig(f'./tcav_scores_histograms/{label}_{layer}_concept_{all_tcav_concept_num[-1]}_TCAV_scores_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()
