import pandas as pd

def parse_image_numbers(start, lines):

    line_one = lines[start][16:-1].split(' ')
    line_two = lines[start+1][1:-1].split(' ')
    image_numbers = line_one + line_two

    image_numbers_ints = []
    for num in image_numbers:
        try:
            image_numbers_ints.append(int(num))
        except:
            pass

    total_image_numbers_ints = image_numbers_ints
    distinct_image_numbers_ints = list(set(image_numbers_ints))

    total_image_numbers_by_label = {}
    for image_num in total_image_numbers_ints:

        label = index_lookup[index_lookup['img_idx'] == image_num]['label'].values[0]

        if label in total_image_numbers_by_label:
            total_image_numbers_by_label[label] += 1
        else:
            total_image_numbers_by_label[label] = 1

    distinct_image_numbers_by_label = {}
    for image_num in distinct_image_numbers_ints:

        label = index_lookup[index_lookup['img_idx'] == image_num]['label'].values[0]

        if label in distinct_image_numbers_by_label:
            distinct_image_numbers_by_label[label] += 1
        else:
            distinct_image_numbers_by_label[label] = 1
    
    return total_image_numbers_by_label, distinct_image_numbers_by_label

if __name__ == '__main__':

    filepath = '../ACE/ACE/concept_discovery_results/mixed_8_cab_results.txt'
    index_lookup_filepath = None
    index_lookup = None

    if index_lookup_filepath:
        index_lookup = pd.read_csv(index_lookup_filepath)
        index_lookup.loc[:,'img_idx'] = index_lookup.index.values

    filename = filepath.split('/')[-1].split('.')[0]

    with open(filepath) as f:
        lines = f.readlines()
    
    labels = []
    concept_labels = []
    highly_common_concept = []
    cond2 = []
    cond3 = []
    concept_is_acceptable = []
    all_total_img_numbers_by_label = []
    all_distinct_img_numbers_by_label = []

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
            concept_is_acceptable_bool = line.split(' ')[-1].strip()
            concept_is_acceptable.append(concept_is_acceptable_bool)
            if index_lookup is not None:
                total_image_numbers_by_label, distinct_image_numbers_by_label = parse_image_numbers(idx+2, lines[2:])
        
        if 'concept_is_acceptable:' in line:
            if index_lookup and total_image_numbers_by_label:
                concept_labels.append(lines[idx+4].split(' ')[-1].strip())
                all_total_img_numbers_by_label.append(total_image_numbers_by_label)
                all_distinct_img_numbers_by_label.append(distinct_image_numbers_by_label)
            else:
                concept_labels.append(None)
                all_total_img_numbers_by_label.append(None)
                all_distinct_img_numbers_by_label.append(None)

    df = pd.DataFrame({
        'labels': labels,
        'concept_labels': concept_labels,
        'highly_common_concept': highly_common_concept,
        'cond2': cond2,
        'cond3': cond3,
        'concept_is_acceptable': concept_is_acceptable,
        'total_img_numbers_by_label': all_total_img_numbers_by_label,
        'distinct_img_numbers_by_label': all_distinct_img_numbers_by_label
    })

    df.to_csv(f'./concept_discovery_results/{filename}.csv', index=False)