import numpy as np
from sklearn import svm, preprocessing, metrics
from imblearn.metrics import geometric_mean_score, specificity_score

class metric_list():

    def __init__(self, gamma, c, num_folders):

        self.num_folders = num_folders
        self.par_a = gamma
        self.par_b = c
        self.num_par_a = len(gamma)
        self.num_par_b = len(c)
        self.Accuracy = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.Precision = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.Recall = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.Specificity = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.G_Mean = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.F_Mean = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))
        self.AUC = np.zeros((self.num_par_a, self.num_par_b, self.num_folders))

    def measure(self, j_gamma, k_c, i_folder, Label_test, Label_predict):
        j = j_gamma
        k = k_c
        i = i_folder
        self.Accuracy[j, k, i] = metrics.accuracy_score(Label_test, Label_predict)
        self.Precision[j, k, i] = metrics.precision_score(Label_test, Label_predict)
        self.Recall[j, k, i] = metrics.recall_score(Label_test, Label_predict)
        self.Specificity[j, k, i] = specificity_score(Label_test, Label_predict)
        self.G_Mean[j, k, i] = geometric_mean_score(Label_test, Label_predict)
        self.F_Mean[j, k, i] = metrics.f1_score(Label_test, Label_predict)

    def auc_measure(self, j_gamma, k_c, i_folder, Label_test, Label_score):
        j = j_gamma
        k = k_c
        i = i_folder
        self.AUC[j, k, i] = metrics.roc_auc_score(Label_test, Label_score)

    def index_find(self):
        Accuracy_t = np.zeros((self.num_par_a, self.num_par_b))
        Precision_t = np.zeros((self.num_par_a, self.num_par_b))
        Recall_t = np.zeros((self.num_par_a, self.num_par_b))
        Specificity_t = np.zeros((self.num_par_a, self.num_par_b))
        G_Mean_t = np.zeros((self.num_par_a, self.num_par_b))
        F_Mean_t = np.zeros((self.num_par_a, self.num_par_b))
        AUC_t = np.zeros((self.num_par_a, self.num_par_b))

        for j in range(self.num_par_a):
            for k in range(self.num_par_b):
                Accuracy_t[j, k] = np.mean(self.Accuracy[j, k, :])
                Precision_t[j, k] = np.mean(self.Precision[j, k, :])
                Recall_t[j, k] = np.mean(self.Recall[j, k, :])
                Specificity_t[j, k] = np.mean(self.Specificity[j, k, :])
                G_Mean_t[j, k] = np.mean(self.G_Mean[j, k, :])
                F_Mean_t[j, k] = np.mean(self.F_Mean[j, k, :])
                AUC_t[j, k] = np.mean(self.AUC[j, k, :])

        par_a = []
        par_b = []
        accuracy = np.max(Accuracy_t)
        accuracy_parameters = np.argwhere(Accuracy_t == accuracy)
        par_a.append(accuracy_parameters[0][0])
        par_b.append(accuracy_parameters[0][1])
        precision = np.max(Precision_t)
        precision_parameters = np.argwhere(Precision_t == precision)
        par_a.append(precision_parameters[0][0])
        par_b.append(precision_parameters[0][1])
        recall = np.max(Recall_t)
        recall_parameters = np.argwhere(Recall_t == recall)
        par_a.append(recall_parameters[0][0])
        par_b.append(recall_parameters[0][1])
        specificity = np.max(Specificity_t)
        specificity_parameters = np.argwhere(Specificity_t == specificity)
        par_a.append(specificity_parameters[0][0])
        par_b.append(specificity_parameters[0][1])
        g_mean = np.max(G_Mean_t)
        g_parameters = np.argwhere(G_Mean_t == g_mean)
        par_a.append(g_parameters[0][0])
        par_b.append(g_parameters[0][1])
        f_mean = np.max(F_Mean_t)
        f_parameters = np.argwhere(F_Mean_t == f_mean)
        par_a.append(f_parameters[0][0])
        par_b.append(f_parameters[0][1])
        auc = np.max(AUC_t)
        auc_parameters = np.argwhere(AUC_t == auc)
        par_a.append(auc_parameters[0][0])
        par_b.append(auc_parameters[0][1])

        return par_a, par_b


    def output(self, name, method, dir):
        Accuracy_t = np.zeros((self.num_par_a, self.num_par_b))
        Precision_t = np.zeros((self.num_par_a, self.num_par_b))
        Recall_t = np.zeros((self.num_par_a, self.num_par_b))
        Specificity_t = np.zeros((self.num_par_a, self.num_par_b))
        G_Mean_t = np.zeros((self.num_par_a, self.num_par_b))
        F_Mean_t = np.zeros((self.num_par_a, self.num_par_b))
        AUC_t = np.zeros((self.num_par_a, self.num_par_b))

        for j in range(self.num_par_a):
            for k in range(self.num_par_b):
                Accuracy_t[j, k] = np.mean(self.Accuracy[j, k, :])
                Precision_t[j, k] = np.mean(self.Precision[j, k, :])
                Recall_t[j, k] = np.mean(self.Recall[j, k, :])
                Specificity_t[j, k] = np.mean(self.Specificity[j, k, :])
                G_Mean_t[j, k] = np.mean(self.G_Mean[j, k, :])
                F_Mean_t[j, k] = np.mean(self.F_Mean[j, k, :])
                AUC_t[j, k] = np.mean(self.AUC[j, k, :])

        file_wirte = name
        with open(file_wirte, 'a') as w:
            # if First_line:
            #    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
            #    first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
            #    w.write(first_line)
            #    First_line = False

            line = dir + '\t' + method + '\t'
            accuracy = np.max(Accuracy_t)
            accuracy_parameters = np.argwhere(Accuracy_t == accuracy)
            line += str(accuracy) + '\t' + str('%.3f' % self.par_a[accuracy_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[accuracy_parameters[0][1]]) + '\t'

            precision = np.max(Precision_t)
            precision_parameters = np.argwhere(Precision_t == precision)
            line += str(precision) + '\t' + str('%.3f' % self.par_a[precision_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[precision_parameters[0][1]]) + '\t'

            recall = np.max(Recall_t)
            recall_parameters = np.argwhere(Recall_t == recall)
            line += str(recall) + '\t' + str('%.3f' % self.par_a[recall_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[recall_parameters[0][1]]) + '\t'

            specificity = np.max(Specificity_t)
            specificity_parameters = np.argwhere(Specificity_t == specificity)
            line += str(specificity) + '\t' + str('%.3f' % self.par_a[specificity_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[specificity_parameters[0][1]]) + '\t'

            g_mean = np.max(G_Mean_t)
            g_parameters = np.argwhere(G_Mean_t == g_mean)
            line += str(g_mean) + '\t' + str('%.3f' % self.par_a[g_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[g_parameters[0][1]]) + '\t'

            f_mean = np.max(F_Mean_t)
            f_parameters = np.argwhere(F_Mean_t == f_mean)
            line += str(f_mean) + '\t' + str('%.3f' % self.par_a[f_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[f_parameters[0][1]]) + '\t'

            auc = np.max(AUC_t)
            auc_parameters = np.argwhere(AUC_t == auc)
            line += str(auc) + '\t' + str('%.3f' % self.par_a[auc_parameters[0][0]]) + \
                    ',' + str('%.3f' % self.par_b[auc_parameters[0][1]]) + '\n'

            w.write(line)

    def output_b(self, name, method, dir, par_a, par_b):
        accuracy = np.mean(self.Accuracy[par_a[0], par_b[0], :])
        precision = np.mean(self.Precision[par_a[1], par_b[1], :])
        recall = np.mean(self.Recall[par_a[2], par_b[2], :])
        specificity = np.mean(self.Specificity[par_a[3], par_b[3], :])
        g_mean = np.mean(self.G_Mean[par_a[4], par_b[4], :])
        f_mean = np.mean(self.F_Mean[par_a[5], par_b[5], :])
        auc = np.mean(self.AUC[par_a[6], par_b[6], :])

        file_wirte = name
        with open(file_wirte, 'a') as w:
            # if First_line:
            #    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
            #    first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
            #    w.write(first_line)
            #    First_line = False

            line = dir + '\t' + method + '\t'
            line += str(accuracy) + '\t' + str('%.3f' % self.par_a[par_a[0]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[0]]) + '\t'

            line += str(precision) + '\t' + str('%.3f' % self.par_a[par_a[1]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[1]]) + '\t'

            line += str(recall) + '\t' + str('%.3f' % self.par_a[par_a[2]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[2]]) + '\t'

            line += str(specificity) + '\t' + str('%.3f' % self.par_a[par_a[3]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[3]]) + '\t'

            line += str(g_mean) + '\t' + str('%.3f' % self.par_a[par_a[4]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[4]]) + '\t'

            line += str(f_mean) + '\t' + str('%.3f' % self.par_a[par_a[5]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[5]]) + '\t'

            line += str(auc) + '\t' + str('%.3f' % self.par_a[par_a[6]]) + \
                    ',' + str('%.3f' % self.par_b[par_b[6]]) + '\n'

            w.write(line)

