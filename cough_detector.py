from sklearn import svm
from sklearn import metrics
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import optparse
import csv
import os
import sys
import codecs

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

"""Detects cough on given audio features"""
class CoughDetector:
    def __init__(self, target_subject, num_subjects=12, verbose=False):
        self.target_subject = target_subject
        self.verbose = verbose
        self.num_subjects = num_subjects
        self.n_features = 34
        self.X_test = []
        self.X_train = []
        self.y_train = []
        self.y_test = []
        self.features = []
        self.labels = []
        self.has_model = False
        self.class_names = {"sintos": 0, "contos": 1}
        self.filepath = os.path.dirname(os.path.abspath(__file__))
        print bcolors.HEADER + "Leaving {}. subject out".format(self.target_subject) + bcolors.ENDC
        """Load model if any, otherwise train it"""
        try:
            # self.model = load('models/model{}.joblib'.format(self.target_subject))
            self.model = load('models/model.joblib'.format(self.target_subject))
            print bcolors.OKGREEN + "Model was loaded" + bcolors.ENDC
            self.has_model = True
            self.extract_features_from_audio()     # Import and extract features from corresponding target audio
        except IOError:
            print bcolors.FAIL + "Model was not found" + bcolors.ENDC
            """Create Model"""
            self.model = svm.SVC(kernel='rbf') # Radial Basis Function
            """Load feature dataset and target list from txt file"""
            # print bcolors.WARNING + "Loading feature dataset from txt file..." + bcolors.ENDC
            # self.import_feature_dataset()
            """Compute feature dataset using pyAudioAnalysis applied to audio files"""
            print bcolors.WARNING + "Extracting features from audio files (this may take a while)..." + bcolors.ENDC
            self.extract_features_from_audio()
            """Training model from audio files (by default)"""
            self.train_model()
            dump(self.model, 'models/model{}.joblib'.format(self.target_subject))
        """Predict response for test dataset"""
        print bcolors.WARNING + "Predicting by using trained model..." + bcolors.ENDC
        y_target = self.model.predict(self.X_test)
        """Evaluate the model"""
        accuracy = metrics.accuracy_score(self.y_test, y_target)
        print bcolors.OKBLUE + "Accuracy: {}".format(accuracy) + bcolors.ENDC
        """Counter counter"""
        threshold = 0.05   # seconds
        threshold = int(threshold*(len(self.X_test)-1)/self.duration)  # Transform to number of frames
        count = self.count_coughs(y_target, threshold)
        if count > 0:
            print bcolors.HEADER + "Somebody coughed!" + bcolors.ENDC
        else:
            print bcolors.HEADER + "There is no cough here!" + bcolors.ENDC

    """Train model either from given features or from audio files"""
    def train_model(self):
        """Split data with given proportion"""
        # if self.verbose: print bcolors.WARNING + "Splitting data for training..." + bcolors.ENDC
        # self.X_train, self.X_test = self.split_data(self.features, 0.7)
        # self.y_train, self.y_test = self.split_data(self.labels, 0.7)
        if self.verbose:
            print "Length of splitted datasets: {} & {}".format(len(self.X_train), len(self.X_test))
            print "Length of splitted targets: {} & {}".format(len(self.y_train), len(self.y_test))
            print "No. features: {}".format(len(self.X_train[0]))
        """Train model from given feature dataset"""
        if self.verbose: print bcolors.WARNING + "Training model..." + bcolors.ENDC
        self.model.fit(self.X_train, self.y_train)

    """Create feature dataset from given file"""
    def import_feature_dataset(self):
        f_names = ['variance', 'energy', 'centroid', 'frequency']   # Feature names
        # Load data from csv files
        try:
            # Alternative approach due to atypical file"""
            # reader = csv.DictReader(codecs.open('{}/features{}.txt'.format(self.filepath, sub), 'rU', 'utf-16'), delimiter='\t')
            for sub in range(1, self.num_subjects+1):
                """Conventional .txt reading"""
                with open('{}/features{}.txt'.format(self.filepath, sub), mode='rb') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter='\t')
                    for row in reader:
                        feature = []
                        for name in f_names:
                            feature.append(row[name])
                        if sub == self.target_subject:
                            self.X_test.append(feature)
                            self.y_test.append(self.class_names[row['target']])
                        else:
                            self.X_train.append(feature)
                            self.y_train.append(self.class_names[row['target']])
        except IOError as e:
            print("No dataset file found: {}".format(e))
            sys.stdout.close()

    """Extract features from audio files"""
    def extract_features_from_audio(self):
        for sub in range(1, self.num_subjects+1):
            if (not self.has_model or (self.has_model and sub == self.target_subject)):
                """Audio importation"""
                if self.verbose: print bcolors.WARNING + "\nReading audio of {}. subject".format(sub) + bcolors.ENDC
                [Fs, x] = audioBasicIO.read_audio_file("{}/raw_data/audio{}.wav".format(self.filepath, sub))
                self.duration = len(x)/float(Fs)
                if self.verbose:
                    print "Sampling frequency: {}".format(Fs)
                    print "Audio self.duration: {} s".format(self.duration)
                """Extract features with a frame size 10 msec and frame step of 5 msec (50% overlap)"""
                if self.verbose: print bcolors.WARNING + "Extracting features..." + bcolors.ENDC
                frame_size = 0.01   # seconds
                overlap = 0.5
                F, f_names = ShortTermFeatures.feature_extraction(x, Fs, frame_size*Fs, frame_size*overlap*Fs)   # 34 features
                """Feature plots"""
                # plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
                # plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
                """Creating feature dataset with the structure needed for scikit-learn"""
                if self.verbose: print bcolors.WARNING + "Restructuring feature dataset..." + bcolors.ENDC
                for f in range(len(F[0])):
                    frame = []
                    for i in range(self.n_features):
                        frame.append(F[i][f])
                    # self.features.append(frame)   # Without cross-validation and further data splitting
                    if sub == self.target_subject: self.X_test.append(frame)
                    else: self.X_train.append(frame)
                """Retrieving manual timestamps from txt file"""
                if self.verbose: print bcolors.WARNING + "Reading timestamps file..." + bcolors.ENDC
                timestamps = []
                try:
                    with open('{}/raw_data/timestamps{}.txt'.format(self.filepath, sub), mode='rb') as csvfile:
                        reader = csv.reader(csvfile, delimiter='\t')
                        for row in reader:
                            ts = []
                            ts.append(float(row[0]))
                            ts.append(float(row[1]))
                            timestamps.append(ts)
                except IOError as e:
                    print("No timestamp file found: {}".format(e))
                    sys.stdout.close()
                """Fill labels array according to timestamps"""
                if self.verbose: print bcolors.WARNING + "Creating labels array from given timestamps..." + bcolors.ENDC
                labels = np.zeros(len(F[0]))
                for ts in timestamps:
                    start = int(ts[0]*(len(F[0])-1)/self.duration)
                    stop = int(ts[1]*(len(F[0])-1)/self.duration)
                    labels[start:stop+1].fill(1)
                # self.labels.extend(labels)   # Without cross-validation and further data splitting
                if sub == self.target_subject: self.y_test.extend(labels)
                else: self.y_train.extend(labels)

    """Split given array with a given proportion"""
    def split_data(self, data, proportion):
        split_point = int(len(data)*proportion - 1)
        return data[:split_point], data[split_point:]

    """Count how many subarrays of 1s are there in given list"""
    def count_coughs(self, list, threshold):
        frames = 0
        cough_count = 0
        prev = list[0]
        for label in list:
            if label == 1 and prev == 1: frames += 1
            else: frames = 1
            if frames > threshold:
                cough_count += 1
                frames = 0
            prev = label
        return cough_count

def main():
    parser = optparse.OptionParser()
    parser.add_option('-v', '--verbose',
            action="store_true", dest="verbose",
            help="Print verbose", default=False)
    options, args = parser.parse_args()
    target_subject = int(args[0])
    detector = CoughDetector(target_subject, verbose=options.verbose)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("User interruption: {}".format(e))
        sys.stdout.close()
