import pandas as pd

if __name__ == '__main__':

    filepath = '../ACE/ACE/concept_discovery_results/mixed_8_ambulance_and_police_van_results.txt'
    filename = filepath.split('/')[-1].split('.')[0]

    with open(filepath) as f:
        lines = f.readlines()

    labels = []
    highly_common_concept = []
    cond2 = []
    cond3 = []
    concept_is_acceptable = []

    for idx, line in enumerate(lines[2:]):

        if 'label:' in line:
            labels.append(line.split(' ')[-1].strip())
        elif 'highly_common_concept' in line:
            highly_common_concept.append(line.split(' ')[-1].strip())
        elif 'mildly_populated_concept and mildly_common_concept:' in line:
            cond2.append(line.split(' ')[-1].strip())
        elif 'non_common_concept and highly_populated_concept:' in line:
            cond3.append(line.split(' ')[-1].strip())
        elif 'concept_is_acceptable:' in line:
            concept_is_acceptable.append(line.split(' ')[-1].strip())

    df = pd.DataFrame({
        'labels': labels,
        'highly_common_concept': highly_common_concept,
        'cond2': cond2,
        'cond3': cond3,
        'concept_is_acceptable': concept_is_acceptable
    })

    df.to_csv(f'./concept_discovery_results/{filename}.csv', index=False)