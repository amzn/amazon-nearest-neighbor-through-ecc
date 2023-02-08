from src.algos import *
import pandas as pd
from datetime import datetime
import os

from src.algos.ground_truth import GroundTruth
from src.algos.utils.standard_utils import *
from src.algos.factories import QueryFactory, GroundTruthFactory


# import datetime

class Testing:

    @classmethod
    def _calc_recall_thres(cls, *,
                           recall_thres: int,
                           distances,
                           size_nn: int,
                           approx_factor=1) -> float:
        """
        Test the recall of the given algorithm
        :key algo: the algorithm to test for recall.
        :key recall_thres: for each query, the acceptable distance threshold for the output to count for recall.
        :key queries: the query_dict to perform.
        :key size_nn: the number of nearest neighbors to find
        :return ndis: the number of distance query_dict performed in the given input
        :return recall: the recall of the algorithm, as a fraction between 0 and 1.
        """

        recall_vec = cls._calc_recall_vector(recall_thres=recall_thres,
                                             distances=distances,
                                             size_nn=size_nn,
                                             approx_factor=approx_factor)
        recall = np.average(recall_vec)
        return recall

    @classmethod
    def _calc_recall_vector(cls, *,
                            recall_thres: int,
                            distances,
                            size_nn: int,
                            approx_factor=1):

        assert distances.shape[-1] >= size_nn
        relevant_distances = distances[..., :size_nn]

        correct = (relevant_distances <= approx_factor * recall_thres[..., np.newaxis])
        recall_vec = np.average(correct, axis=-1)
        # recall = np.average(recall_vec)
        # ndis = algo.get_ndis()
        # nop = algo.get_nop()
        return recall_vec

    @classmethod
    def calc_algo_recall(cls, algo: Algo, *,
                         datamodel,
                         num_queries,
                         dm_filename,
                         change_param_name,
                         change_func,
                         change_param_values,
                         size_nn_list=[1, 5, 10, 50, 100],
                         alpha_list=np.linspace(1.0, 2.0, num=11),
                         overwrite=False,
                         volatile=False):
        """
        :param algo: the algorithm to test
        :param datamodel: the real_datamodel from which the data is generated
        :param num_queries: the number of query_dict to generate
        :param dm_filename: the filename of the real_datamodel
        :param change_param_name: the name of the parameter to change in each iteration of testing
        :param change_func: the function used to set the new value of change_param_name
        :param change_param_values: the values on which to iterate for the parameter change_param_name
        :return:
        """
        query_class = datamodel.QUERY_CLASS
        query_obj, query_filename = QueryFactory.get_query(query_class,
                                                           dm_filename,
                                                           num_queries=num_queries,
                                                           datamodel=datamodel)

        # query, query_filename = datamodel.get_queries(num_queries=num_queries)

        max_size_nn = max(size_nn_list)
        gt, gt_filename = GroundTruthFactory.get_gt(datamodel=datamodel,
                                                    query_obj=query_obj,
                                                    query_filename=query_filename,
                                                    size_nn=max_size_nn)

        true_distances, true_points = gt.distances, gt.points
        #
        # DBrute, _ = brute.query(query_dict, size_nn)
        # recall_thres = np.max(true_distances, axis=1)

        df_filename = ResultDF.get_dataframe_filename(dm_filename=dm_filename,
                                                      num_queries=num_queries,
                                                      )
        # cls.fill_missing(df_filename)
        skipped = False

        for val in change_param_values:
            lines_to_add = []
            change_func(val)

            param_dict = algo.get_params()
            print_dict = param_dict.copy()
            if print_dict['algo_name'] == 'NoamLSH':
                print_dict['algo_name'] = 'LSH'
            print('TESTING: ' + Algo.name_from_params(param_dict))
            algo_in_df = ResultDF.is_dict_in_dataframe(val_dict=param_dict, df_filename=df_filename)
            if algo_in_df and not volatile and not overwrite:
                skipped = True
                continue

            distances, points = algo.query(datamodel=datamodel,
                                           query_obj=query_obj,
                                           size_nn=max_size_nn)
            avg_ndis = algo.get_ndis() / num_queries
            avg_nop = algo.get_nop() / num_queries

            if overwrite and not volatile:
                ResultDF.purge_algo_from_dataframe(algo=algo, df_filename=df_filename)

            for size_nn in size_nn_list:
                for alpha in alpha_list:
                    recall_thres = true_distances[..., size_nn - 1]
                    recall = Testing._calc_recall_thres(recall_thres=recall_thres,
                                                        distances=distances,
                                                        size_nn=size_nn,
                                                        approx_factor=alpha, )

                    if volatile:
                        print(
                            f'For size_nn={size_nn}, alpha={alpha}, {Algo.name_from_params(algo.get_params())} results:')
                        print(
                            f'finished with {change_param_name}={val}, ndis/query is {avg_ndis}, nop/query is {avg_nop}, recall is {recall}')
                        continue

                    # write results to dataframe
                    result_dict = {
                        'ndis': avg_ndis,
                        'nop': avg_nop,
                        'recall': recall,
                        'alpha': alpha,
                        'size_nn': size_nn,
                    }
                    algo_dict = algo.get_params()
                    line_dict = {**result_dict, **algo_dict}
                    lines_to_add.append(line_dict)
                    # ResultDF.append_result_to_dataframe(result_dict=result_dict, algo=algo, df_filename=df_filename)
            if len(lines_to_add) > 0:
                line_dict = {key: [dicty[key] for dicty in lines_to_add] for key in lines_to_add[0]}
                ResultDF.add_lines_to_dataframe(lines_dict=line_dict, df_filename=df_filename)

            if skipped:
                print(
                    f"calc_algo_recall: WARNING: some results for algo {Algo.name_from_params(algo.get_params())} already exist. Skipped some tests.")
            if volatile:
                print(
                    f"calc_algo_recall: Volatile chosen, results for algo {Algo.name_from_params(algo.get_params())} are not saved.")
            # assert np.array_equal(query_dict['queries'], queries)


class ResultDF:
    RESULTS_PATH = 'Resources/Results/'

    @classmethod
    def add_lines_to_dataframe(cls, *, lines_dict, df_filename):
        df_filepath = cls.RESULTS_PATH + df_filename
        if os.path.exists(df_filepath):
            df = cls.read_dataframe(df_filename=df_filename)
        else:
            df = pd.DataFrame()
        time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        new_lines_df = pd.DataFrame(data=lines_dict)
        new_lines_df['time'] = time
        new_df = pd.concat([df, new_lines_df], ignore_index=True)
        cls.write_dataframe(df=new_df, df_filename=df_filename)

    @classmethod
    def append_result_to_dataframe(cls, *, result_dict, algo, df_filename):
        df_filepath = cls.RESULTS_PATH + df_filename
        if os.path.exists(df_filepath):
            df = cls.read_dataframe(df_filename=df_filename)
        else:
            df = pd.DataFrame()
        time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        row_dict = dict(algo.get_params())
        row_dict.update(result_dict)
        row_dict['time'] = time
        df = df.append(row_dict, ignore_index=True)
        cls.write_dataframe(df=df, df_filename=df_filename)

    @classmethod
    def get_dataframe_lines_matching_algo(cls, *, algo, df):
        algo_params = algo.get_params()
        return cls.get_dataframe_lines_matching_dict(df=df, val_dict=algo_params)

    @classmethod
    def purge_dict_from_all(cls, *, params_dict):
        for filename in os.listdir(cls.RESULTS_PATH):
            if os.path.splitext(filename)[1] == '.csv':
                cls.purge_dict_from_dataframe(params_dict=params_dict, df_filename=filename)

    @classmethod
    def purge_algo_from_dataframe(cls, *, algo, df_filename):
        algo_params = algo.get_params()
        cls.purge_dict_from_dataframe(params_dict=algo_params, df_filename=df_filename)

    @classmethod
    def purge_dict_from_dataframe(cls, *, params_dict, df_filename):
        df = cls.read_dataframe(df_filename=df_filename)
        matches = cls.get_dataframe_lines_matching_dict(df=df, val_dict=params_dict)
        df.drop(matches.index, inplace=True)
        cls.write_dataframe(df=df, df_filename=df_filename)

    @classmethod
    def read_dataframe(cls, *, df_filename):
        df_filepath = cls.RESULTS_PATH + df_filename
        return pd.read_csv(df_filepath, index_col=0)

    @classmethod
    def write_dataframe(cls, *, df, df_filename):
        df_filepath = cls.RESULTS_PATH + df_filename
        return df.to_csv(df_filepath)

    @classmethod
    def get_dataframe_lines_matching_dict(cls, *, df, val_dict):
        for val in val_dict:
            if val not in df.columns:
                return df[0:0]  # empty df
        return df.loc[(df[list(val_dict)] == pd.Series(val_dict)).all(axis=1)]

    @classmethod
    def is_dict_in_dataframe(cls, *, val_dict, df_filename):
        dataframe_filepath = cls.RESULTS_PATH + df_filename
        if not os.path.exists(dataframe_filepath):
            return False
        df = cls.read_dataframe(df_filename=df_filename)
        matches = cls.get_dataframe_lines_matching_dict(val_dict=val_dict, df=df)
        return len(matches) != 0

    @classmethod
    def is_algo_in_dataframe(cls, *, algo, df_filename):
        param_dict = algo.get_params()
        return cls.is_dict_in_dataframe(val_dict=param_dict, df_filename=df_filename)

    @classmethod
    def get_dataframe_filename(cls, *, dm_filename, num_queries):
        file_string = os.path.splitext(dm_filename)[0]
        param_string = f'numquer={num_queries}'
        df_filename_noext = file_string + '___' + param_string

        return df_filename_noext + '.csv'
