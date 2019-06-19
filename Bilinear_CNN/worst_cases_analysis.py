"""this will analyze the worst cases of the results """
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt

def increase_counter(given_map, counter_val, key):
    """increased the given key value by the val """
    val = given_map[key]
    val = val + counter_val
    given_map[key] = val

def show_relevant_info(given_map):
    """will show the classes with non-zero values """
    print('showing relevant info')
    total_mismatches = 0
    for key in given_map:
        val = given_map[key]
        if val != 0:
            print('{0} -> {1}'.format(key, val))
            total_mismatches += val
    return total_mismatches

def draw_mismatch_pie_chart(given_map, total_mismatches):
    """this will draw a pie-chart for easier reference """
    pie_map = {}
    num = 8
    axis = plt.gca()
    color_map = plt.get_cmap('gist_rainbow')
    axis.set_prop_cycle(color=[color_map(1.0 * i / num) for i in range(num)])
    for key in given_map:
        val = given_map[key]
        if val != 0:
            pie_map[key] = (val * 1.0 / total_mismatches ) * 100.00
    plt.pie([float(v) for v in pie_map.values()], labels=[str(k) for k in pie_map.keys()], autopct='%1.1f%%', shadow=False)
    plt.show()

def show_graphs(mismatch_counter, total_counter):
    """this will show the graphs """
    final_matrix = {}
    labels = []
    sorted_matrix = {}
    ordered_matrix = OrderedDict()
    for key in mismatch_counter:
        class_name = key[5:]
        class_name = str(class_name).zfill(2)
        new_name = class_name
        sorted_matrix[new_name] = mismatch_counter[key]
        total_counter[new_name] = total_counter[key]
    labels.clear()
    for key in sorted_matrix:
        labels.append(key)
    labels.sort()
    for key in labels:
        ordered_matrix[key] = sorted_matrix[key]
    labels.clear()
    for key in ordered_matrix:
        total_test_cases = total_counter[key]
        mismatches = ordered_matrix[key]
        accuracy = total_test_cases - mismatches
        final_matrix[key] = (accuracy * 1.0 / total_test_cases) * 100.00
        labels.append(key)

    #print(mismatch_counter)
    #print(final_matrix)
    plt.bar(range(len(final_matrix)), final_matrix.values(), align='center')
    axis = plt.gca()
    axis.set_xticks(range(len(final_matrix)))
    axis.set_xticklabels(labels, rotation=0)
    plt.show()

def process_file(filename):
    """this will process the results for the filename """
    with open(filename, 'r') as reader:
        # ignore the header
        next(reader)
        csv_reader = csv.reader(reader, delimiter=',')
        mismatch_counter = {}
        total_counter = {}
        for row in csv_reader:
            name = str(row[0]).strip()
            prediction = str(row[1]).strip()
            class_name = name.split('\\')[0].strip()
            if class_name in total_counter:
                increase_counter(total_counter, 1, class_name)
            else:
                total_counter[class_name] = 1
            if class_name != prediction:
                #we have a mismatch
                if class_name in mismatch_counter:
                    increase_counter(mismatch_counter, 1, class_name)
                else:
                    mismatch_counter[class_name] = 1
            else:
                if class_name in mismatch_counter:
                    increase_counter(mismatch_counter, 0, class_name)
                else:
                    mismatch_counter[class_name] = 0
    mismatches = show_relevant_info(mismatch_counter)
    print(mismatches)
    print(total_counter)
    #draw_mismatch_pie_chart(mismatch_counter, mismatches)
    show_graphs(mismatch_counter, total_counter)

def run():
    """the main method to run """
    #process_file('results.csv')
    process_file('results_GAN.csv')


if __name__ == "__main__":
    run()
