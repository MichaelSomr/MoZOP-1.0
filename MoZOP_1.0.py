import os.path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def main(site_name, path, subset_size=15, subset_spacing=15, tolerance=5, graphical_output=False):

    # read roiData.dat to get geometry of observed poles
    segments_name, segments_x, segments_y = [], [], []
    with open('roiData.dat') as fp:
        for line in fp:
            line = line.rstrip()
            arr = line.split('\t')
            segments_name.append(arr[0])
            segments_x.append(np.array([int(arr[1]), int(arr[2])]))
            segments_y.append(np.array([int(arr[3]), int(arr[4])]))

    # create output folder
    output_dir = path + '/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get image names
    valid_images = ['.jpg', '.png']
    paths = os.listdir(path)
    paths.sort()
    ref_subsets = []

    # count images
    n_images = 0
    for f in paths:
        file_name = path + '/' + f
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        n_images += 1

    n_image = 0
    counter = 0
    results_file = open('%s/output/results_%s.txt' % (path, site_name), 'w')
    # loop through images and find correspondence with the reference one
    for f in paths:
        file_name = path + '/' + f
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        n_image += 1
        print('Processing file %s (%d of %d).' % (file_name, n_image, n_images))

        cur_image_rgb = mpimg.imread(file_name)
        cur_image = np.dot(cur_image_rgb[..., :3], [0.299, 0.587, 0.114])

        if graphical_output:
            fig = plt.figure()
            plt.imshow(cur_image, cmap='gray')

        cur_states = []
        percentages = []
        for j in range(len(segments_name)):

            if counter == 0:
                ref_subsets.append([])
            cur_states.append([])
            segment_name = segments_name[j]
            segment_x = segments_x[j]
            segment_y = segments_y[j]

            n_subsets = int(np.sqrt((segment_x[1] - segment_x[0]) ** 2 +
                                    (segment_y[1] - segment_y[0]) ** 2) / subset_spacing)
            xs, ys = [0.0] * (n_subsets + 1), [0.0] * (n_subsets + 1)
            for i in range(n_subsets + 1):
                xs[i] = np.ceil(segment_x[0] + (segment_x[1] - segment_x[0]) / n_subsets * i)
                ys[i] = np.ceil(segment_y[0] + (segment_y[1] - segment_y[0]) / n_subsets * i)

            hs = 0.5 * subset_size
            for i in range(n_subsets):
                x_coords = (xs[i] - hs, xs[i] + hs, xs[i] + hs, xs[i] - hs, xs[i] - hs)
                y_coords = (ys[i] - hs, ys[i] - hs, ys[i] + hs, ys[i] + hs, ys[i] - hs)
                plt.plot(x_coords, y_coords, '-r')

            for i in range(n_subsets):
                cur_subset = cur_image[int(ys[i] - hs): int(ys[i] + hs), int(xs[i] - hs): int(xs[i] + hs)]

                # indicators
                x_coords = (xs[i] - hs, xs[i] + hs, xs[i] + hs, xs[i] - hs, xs[i] - hs)
                y_coords = (ys[i] - hs, ys[i] - hs, ys[i] + hs, ys[i] + hs, ys[i] - hs)

                if counter == 0:
                    if graphical_output:
                        plt.plot(x_coords, y_coords, '-g', linewidth=2)
                    ref_subsets[j].append(cur_subset)
                    cur_states[j].append(0)
                else:
                    sim, diff = structural_similarity(cur_subset, ref_subsets[j][i], full=True)
                    if sim > (1.0 - tolerance / 100):
                        if graphical_output:
                            plt.plot(x_coords, y_coords, '-g', linewidth=2)
                        cur_states[j].append(0)
                    else:
                        if graphical_output:
                            plt.plot(x_coords, y_coords, '-r', linewidth=2)
                        cur_states[j].append(1)

            # calculate percentage of covering
            percentage = sum(cur_states[j]) / len(cur_states[j]) * 100
            percentages.append(percentage)
            min_y = min(segment_y)
            x_at_min_y = segment_x[np.where(segment_y == min_y)]
            if graphical_output:
                ax = plt.gca()
                plt.text(x_at_min_y, min_y - 20, '%s: %.1f %%' % (segment_name, percentage), horizontalalignment='center',
                         verticalalignment='bottom', fontsize=25, color='red', bbox=dict(facecolor='white', alpha=0.5,
                                                                                         edgecolor='red', pad=10.0))

        # print output
        if counter == 0:
            # results_file.write('%s\n---\n' % site_name)
            names = ''
            for name in segments_name:
                names += '%s\t' % name
            names.rstrip('\t')
            results_file.write(names + '\n')
        cur_vals = ''
        for val in percentages:
            cur_vals += '%.1f\t' % val
        cur_vals.rstrip('\t')
        results_file.write(cur_vals + '\n')

        counter += 1

        if graphical_output:
            plt.draw()
            plt.tight_layout()
            fig.set_size_inches(30., 18.)
            plt.savefig(path + '/output/%s-%03d.jpg' % (site_name, n_image), dpi=50)
            plt.close(fig)

    results_file.close()


if __name__== "__main__":

    tolerance = 5  # tolerated change in texture within a subset [%] (to recognize if there is a pole cover by debris)
    subset_size = 15  # size of "control windows" (= subsets) in which the changes in texture are evaluated
    subset_spacing = 15  # spacing of subsets
    path = 'test-images'  # path to loaded images and future output path (path/output)
    site_name = 'Zbraslav_test'
    graphical_output = True

    main(site_name, path, subset_size, subset_spacing, tolerance, graphical_output)
