"""This will plot the histograms from the Bilinear CNN model """
import matplotlib.pyplot as plt

PATHS = './model/save/histograms/'

FIRST_STEP = PATHS + '20190412072106/'
SECOND_STEP = PATHS + '20190416112753/'

ACC_FILE_NAME = '_cat_acc.txt'
LOSS_FILE_NAME = '_loss.txt'

FIRST_ACC = []
FIRST_ACC_INDEX = []
FIRST_LOSS = []
FIRST_LOSS_INDEX = []
SECOND_ACC = []
SECOND_ACC_INDEX = []
SECOND_LOSS = []
SECOND_LOSS_INDEX = []

def read_first_step_histogram():
    """this will read the histogram of the first step """
    acc_file = FIRST_STEP + ACC_FILE_NAME
    index = 0
    with open(acc_file, 'r') as reader:
        for line in reader:
            curr_val = float(line) * 100.00
            index += 1
            FIRST_ACC.append(curr_val)
            FIRST_ACC_INDEX.append(index)
    loss_file = FIRST_STEP + LOSS_FILE_NAME
    index = 0
    with open(loss_file, 'r') as reader:
        for line in reader:
            curr_val = float(line) * 100.00
            index += 1
            FIRST_LOSS.append(curr_val)
            FIRST_LOSS_INDEX.append(index)
    print('Read first step histogram')

def read_second_step_histogram():
    """this will read the histogram of the second step """
    acc_file = SECOND_STEP + ACC_FILE_NAME
    index = 0
    with open(acc_file, 'r') as reader:
        for line in reader:
            curr_val = float(line) * 100.00
            index += 1
            SECOND_ACC.append(curr_val)
            SECOND_ACC_INDEX.append(index)
    loss_file = SECOND_STEP + LOSS_FILE_NAME
    index = 0
    with open(loss_file, 'r') as reader:
        for line in reader:
            curr_val = float(line) * 100.00
            index += 1
            SECOND_LOSS.append(curr_val)
            SECOND_LOSS_INDEX.append(index)
    print('Read second step histogram')

def show_graphs():
    """this will show the graphs """
    plt.title('First step and second step histograms')
    # plotting the points
    plt.figure(1)
    plt.subplot(211)
    plt.grid(True)
    plt.xlabel('Units')
    plt.ylabel("First Step Acc")
    plt.plot(FIRST_ACC_INDEX, FIRST_ACC, label="First step acc")

    plt.subplot(212)
    plt.grid(True)
    plt.xlabel("Units")
    plt.ylabel("First step Loss")
    plt.plot(FIRST_LOSS_INDEX, FIRST_LOSS, label="First step loss")

    plt.figure(2)
    plt.subplot(311)
    plt.grid(True)
    plt.xlabel('Units')
    plt.ylabel("Second Step Acc")
    plt.plot(SECOND_ACC_INDEX, SECOND_ACC, label="Second step acc")

    plt.subplot(312)
    plt.grid(True)
    plt.xlabel("Units")
    plt.ylabel("Second step Loss")
    plt.plot(SECOND_LOSS_INDEX, SECOND_LOSS, label="Second step loss")

    plt.show()

def run():
    """the main method to run """
    read_first_step_histogram()
    read_second_step_histogram()
    show_graphs()

if __name__ == "__main__":
    run()
