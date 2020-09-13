from glob import glob

max_count_by_img_class = {}
for img in glob('./occluded_images/**/*'):

    img_class = img.split('/')[-2]
    img_file = '/'.join(img.split('/')[-2:])

    if len(glob(f'./occluded_images/{img_file}/mask_dim_50/*')) == 1000:
        if img_class in max_count_by_img_class:
            max_count_by_img_class[img_class] += 1
        else:
            max_count_by_img_class[img_class] = 1

print(max(max_count_by_img_class, key=max_count_by_img_class.get))
print(max_count_by_img_class[max(max_count_by_img_class, key=max_count_by_img_class.get)])