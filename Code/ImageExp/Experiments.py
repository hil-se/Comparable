import pandas as pd

import Classification as cl
import DataProcessing as dp

num_comp = 1
num_img = 5500

iterations = 5


def experiment(dataName="FaceImage", height=250, width=250, col='Average'):
    final_results = []
    final_results_reg = []
    for i in range(iterations):
        print("Iteration:", i + 1, "/", iterations)
        print("Generating data...")
        training_data, testing_data, testList, dataList, full_len, test_len, protected, train, test, protected_ts = dp.processData(
            h=height, w=width, col=col, num_comp=num_comp, num_img=num_img)
        print("Data generated.")

        # mse, r2, p_coef, p_value, s_coef, s_value, MI, r_sep, accuracy_r, f1_r, precision_r, recall_r = cl.regressionExperiment(
        #     train_val=train,
        #     test=test,
        #     comp_test=testing_data,
        #     height=height,
        #     width=width, col=col)

        recall, precision, f1, acc, AOD, spearmanr, sp_pvalue, pearsonr, p_pvalue, MI = cl.comparabilityExperiment(
            dataName="FaceImage",
            train_val=training_data,
            test=testing_data,
            testList=testList,
            dataList=dataList,
            height=height,
            width=width,
            protected=protected,
            protected_ts=protected_ts)
        # result_reg = {"Full data size": full_len, "Testing data size": test_len,
        #               "MSE": mse, "R2": r2, "P Coef": p_coef,
        #               "P Value": p_value, "SP Coef": s_coef,
        #               "SP Value": s_value, "MI": MI, "R_sep": r_sep, "Recall": recall_r, "Precision": precision_r,
        #               "F1": f1_r, "Accuracy": accuracy_r}

        result = {"Full data size": full_len, "Testing data size": test_len,
                  "Recall": recall, "Precision": precision, "F1": f1, "Accuracy": acc,
                  "AOD": AOD, "Spearman's rank correlation": spearmanr, "SP value": sp_pvalue,
                  "Pearson's rank correlation": pearsonr, "P value": p_pvalue, "MI": MI}
        # print(result)
        final_results.append(result)
        # final_results_reg.append(result_reg)
    final_results = pd.DataFrame(final_results)
    # final_results_reg = pd.DataFrame(final_results_reg)
    print("\n*******************************************")
    print(dataName)
    # print(final_results.mean(axis=0))
    print("*******************************************\n")
    filepath = "../../Results/" + dataName + " Shared Encoder_" + col + "_" + str(num_img) + "_" + str(
        num_comp) + ".csv"
    final_results.to_csv(filepath, index=False)
    # final_results_reg.to_csv("../../Results/" + dataName + " Reg_" + col + "_" + str(num_img) + ".csv", index=False)


experiment(dataName="FaceImage", col='Average', height=250, width=250)
