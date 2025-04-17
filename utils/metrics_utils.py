from transformers import EvalPrediction
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from scipy.stats import pearsonr
from typing import Dict
import numpy as np
import pandas as pd
from utils.constant_utils import * 


class MyComputeMetrics:
    def __init__(self, args):
        """
        Initialize MyComputeMetrics instance with provided arguments.

        Args:
            args: Arguments containing dataset_name, class_type, and label_type.
        """
        self.args = args
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.run_name = args.wandb_run_name
        self.class_type = args.class_type
        self.label_type = args.label_type 
        self.modality = args.modalities

        self.dataset_label_mapping = {
            'MELD': MELD_EMOTION_MAPPING,   # 7 classes
            'IEMOCAP': IEMOCAP_EMOTION_MAPPING_6_CLASS, # 6 classes
            'sentiment': SENTIMENT_MAPPING, # 3 classes
        }

        if self.label_type == 'discrete':
            self.target_names, self.label_names = self.get_dataset_labels()

    def get_dataset_labels(self):
        """
        Get target names and label names based on class type and dataset name.

        Returns:
            Tuple containing target names and label names.
        """
        if self.class_type == '4_ways':
            dataset_label = IEMOCAP_EMOTION_MAPPING_4_CLASS

        elif self.class_type == 'sentiment':
            dataset_label = SENTIMENT_MAPPING

        else: 
            dataset_label = self.dataset_label_mapping.get(self.dataset_name, SENTIMENT_MAPPING)
        return list(dataset_label.keys()), list(dataset_label.values())
    
    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def transfer_results_to_df(self, results: Dict):
        df = pd.DataFrame(results, index=[0])
        df.insert(0, 'dataset', self.dataset_name)
        df.insert(1, 'model', self.model_name)
        df.insert(2, 'modality', self.modality)
        df.insert(3, 'version', self.run_name)
        return df

    def get_classification_report(self, p: EvalPrediction):
        """
        General classification report.

        Args:
            p: EvalPrediction object.

        Returns:
            Dictionary containing classification report metrics.
        """
        preds = np.argmax(p.predictions, axis=1)
        report = classification_report(p.label_ids, preds, target_names=self.target_names, labels=self.label_names, output_dict=True)
        acc_per_class = {k: report[k]['precision'] for k in self.target_names}
        f1_per_class = {k: report[k]['f1-score'] for k in self.target_names}
        weighted_f1 = report['weighted avg']['f1-score']
        try:
            accuracy = report['accuracy']
        except KeyError:
            accuracy = accuracy_score(p.label_ids, preds)

        eval_stats = {}
        for emotion in self.target_names:
            eval_stats[f'{emotion} acc'] = round(100 * acc_per_class[emotion], 2)
            eval_stats[f'{emotion} f1'] = round(100 * f1_per_class[emotion], 2)
        eval_stats['F1'] = round(100 * weighted_f1, 2)
        eval_stats['Accuracy'] = round(100 * accuracy, 2)     
        return eval_stats

    def get_mosi_classification_report(self, p: EvalPrediction):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2   
        }
        """
        y_pred = p.predictions
        y_true = p.label_ids

        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        # two classes 
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        eval_results = {
            "Acc_3": round(100 * Mult_acc_3, 2),
            "F1_3": round(100 * F1_score_3, 2),
            "Has0_acc_2":  round(100 * Has0_acc_2, 2),
            "F1": round(100 * Has0_F1_score, 2),
            "Non0_acc_2":  round(100 * Non0_acc_2, 2),
            "Non0_F1": round(100 * Non0_F1_score, 2),
        }
        return eval_results

    def get_regression_report(self, p: EvalPrediction):
        """
        General regression report.

        Args:
            p: EvalPrediction object.

        Returns:
            Dictionary containing regression report metrics.
        """
        preds = p.predictions.flatten()
        targets = p.label_ids.flatten()
        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds)
        corr, _ = pearsonr(targets, preds)
        return {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Correlation": round(corr, 4),
        }
    
    def get_sims_regression_report(self, p: EvalPrediction):
        y_pred = p.predictions
        y_true = p.label_ids
        test_preds = np.reshape(y_pred, -1) 
        test_truth = np.reshape(y_true, -1)
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "acc_5": round(100 * mult_a5, 2),
            "acc_3": round(100 * mult_a3, 2),
            "acc_2": round(100 * mult_a2, 2),
            "F1": round(100 * f_score, 2),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4), # Correlation Coefficient
        }
        return eval_results

    def get_mosei_regression_report(self, p: EvalPrediction):
        y_pred = p.predictions # (batch_size,1)
        y_true = p.label_ids    # (batch_size,)
        test_preds = np.reshape(y_pred, -1) 
        test_truth = np.reshape(y_true, -1)

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "acc_7": round(100 * mult_a7, 2),
            "acc_5": round(100 * mult_a5, 2),
            "Has0_acc_2":  round(100 * acc2, 2),
            "Has0_F1": round(100 * f_score, 2),
            "Non0_acc_2":  round(100 * non_zeros_acc2, 2),
            "Non0_F1": round(100 * non_zeros_f1_score, 2),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def __call__(self, p: EvalPrediction):
        return self.get_eval_stat(p)

    def get_eval_stat(self, p: EvalPrediction):
        regression_eval_mapping = {
            'CMU_MOSEI': self.get_mosei_regression_report,
            'CMU_MOSI': self.get_mosei_regression_report,
            'CH_SIMS': self.get_sims_regression_report,
            'CH_SIMS_v2': self.get_sims_regression_report,
        }

        classification_eval_mapping = {
            'CMU_MOSEI': self.get_mosi_classification_report,
            'CMU_MOSI': self.get_mosi_classification_report,
            'CH_SIMS': self.get_mosi_classification_report,
            'MELD': self.get_classification_report,
            'IEMOCAP': self.get_classification_report,
        }


        if self.label_type == 'discrete': # classification
            return classification_eval_mapping.get(self.dataset_name, self.get_classification_report)(p)
           
        elif self.label_type == 'continuous': # regression
            return regression_eval_mapping.get(self.dataset_name, self.get_regression_report)(p)