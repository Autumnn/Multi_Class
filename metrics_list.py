import numpy as np
from sklearn import metrics
from imblearn.metrics import geometric_mean_score, specificity_score


class MetricList:

    def __init__(self, num_folders):

        self.num_folders = num_folders
        self.Accuracy = np.zeros(self.num_folders)
        self.Precision = np.zeros(self.num_folders)
        self.Recall = np.zeros(self.num_folders)
        self.Specificity = np.zeros(self.num_folders)
        self.G_Mean = np.zeros(self.num_folders)
        self.F_Mean = np.zeros(self.num_folders)
#        self.AUC = np.zeros(self.num_folders)

    def measure(self, i_folder, Label_test, Label_predict, average):
        i = i_folder
        self.Accuracy[i] = metrics.accuracy_score(Label_test, Label_predict)
#        print(self.Accuracy)
        self.Precision[i] = metrics.precision_score(Label_test, Label_predict, average=average)
#        print(self.Precision)
        self.Recall[i] = metrics.recall_score(Label_test, Label_predict, average=average)
#        print(self.Recall)
        self.Specificity[i] = specificity_score(Label_test, Label_predict, average=average)
#        print(self.Specificity)
        self.G_Mean[i] = geometric_mean_score(Label_test, Label_predict, average=average)
#        print(self.G_Mean)
        self.F_Mean[i] = metrics.f1_score(Label_test, Label_predict, average=average)
#        print(self.F_Mean)

#    def auc_measure(self, i_folder, Label_test, Label_score, average):
#        i = i_folder
#        self.AUC[i] = metrics.roc_auc_score(Label_test, Label_score, average=average)

    def mean_accuracy(self):
        return np.mean(self.Accuracy[:])

    def mean_G(self):
        return np.mean(self.G_Mean[:])

    def mean_F(self):
        return np.mean(self.F_Mean[:])

    def output(self, name, method, dir):
        accuracy = np.mean(self.Accuracy[:])
        precision = np.mean(self.Precision[:])
        recall = np.mean(self.Recall[:])
        specificity = np.mean(self.Specificity[:])
        g_mean = np.mean(self.G_Mean[:])
        f_mean = np.mean(self.F_Mean[:])
#        auc = np.mean(self.AUC)

        file_wirte = name
        with open(file_wirte, 'a') as w:
            # if First_line:
            #    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
            #    first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
            #    w.write(first_line)
            #    First_line = False

            line = dir + '\t' + method
            line += '\t' + str(accuracy)
            line += '\t' + str(precision)
            line += '\t' + str(recall)
            line += '\t' + str(specificity)
            line += '\t' + str(g_mean)
            line += '\t' + str(f_mean)
#            line += '\t' + str(auc)
            line += '\n'
            w.write(line)
