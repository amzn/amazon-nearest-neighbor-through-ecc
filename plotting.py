import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import scipy.spatial
from src.algos import *
from src.testing import ResultDF


class Plotting:

    @classmethod
    def plot_recall_ndis(cls, *,
                         df,
                         title=None,
                         algo_param_list,
                         legend=True,
                         pretty=False,
                         color=None,
                         condition_list=[],
                         x_start=None,
                         x_end=None,
                         use_dashes=False,
                         use_markers=False):

        if pretty:
            first_algo = algo_param_list[0]
            common_params = first_algo.copy()
            del common_params['algo_name']
            diff_params = set()

            for algo_param in algo_param_list:
                pruned_common_params = common_params.copy()
                for key in common_params:
                    if key not in algo_param or algo_param[key] != common_params[key]:
                        del pruned_common_params[key]
                        diff_params.add(key)
                common_params = pruned_common_params
                for key in algo_param:
                    if key not in common_params:
                        diff_params.add(key)

            pretty_keys = {
                'algo_name': 'algo_name',
                'size_nn': 'size_nn',
                'embedding_class': 'emb_class',
                'num_bits': 'nbit',
                'num_tables': 'ntable',
                'code_dim': 'cdim',
                'prrot_iters': 'emb_iters',
                'seed': 'seed',
                'alpha': '$\\alpha$'

            }
            pretty_names = {
                'NoamLSH': 'LSH',
                'IVF': 'IVF',
                'FaissLSH': 'FaissIBMH',
                'PCNN': 'PCNN'
            }

        if use_dashes:
            dash_dict = {
                'NoamLSH': 'dashed',
                'FaissLSH': 'dashed',
                'PCNN': 'solid',
                'IVF': 'dotted'
            }

        if use_markers:
            markers = 'osv^x*Dph'
        num_drawn = 0
        for params in algo_param_list:
            matching_lines = ResultDF.get_dataframe_lines_matching_dict(df=df, val_dict=params)
            if len(matching_lines) == 0:
                continue
            matching_lines = matching_lines[matching_lines['recall'] > 0]
            if len(matching_lines) == 0:
                continue
            to_add = True
            for condition in condition_list:
                matching_lines = matching_lines[condition(matching_lines)]
            if not to_add:
                continue
            inds = matching_lines['ndis'].values.argsort()
            ndis_vals = matching_lines['ndis'].iloc[inds]
            recall_vals = matching_lines['recall'].iloc[inds]

            if use_dashes:
                style = dash_dict[params['algo_name']]
            else:
                style = 'solid'

            if use_markers:
                marker = markers[num_drawn]
            else:
                marker = 'o'

            if pretty:
                diff_keys = (params.keys() & diff_params)
                params['algo_name'] = pretty_names[params['algo_name']]
                params = {pretty_keys[k]: params[k] for k in diff_keys}

            if color is None:
                plt.plot(ndis_vals, recall_vals, linestyle=style, marker=marker, label=Algo.name_from_params(params))
            else:
                plt.plot(ndis_vals, recall_vals, linestyle=style, marker=marker, color=color,
                         label=Algo.name_from_params(params))
            num_drawn += 1

        plt.xlabel('distance computations per query', fontsize=20)
        plt.ylabel('recall', fontsize=20)
        plt.xscale('log')

        if pretty:
            if title is not None:
                plt.suptitle(title)
            # delete problem parameters
            if 'alpha' in common_params:
                del common_params['alpha']
            if 'size_nn' in common_params:
                del common_params['size_nn']
            common_param_string = ', '.join({f'{pretty_keys[k]}={common_params[k]}' for k in common_params})

            plt.title(f'(common parameters: {common_param_string})', fontsize=10)
        else:
            plt.suptitle(title, fontsize=15)
        # plt.legend()
        if legend:
            plt.legend(loc='best', prop={'size': 12})
        # plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", prop={'size': 6}, borderaxespad=0)
        # plt.ylim(0.6, 1)
        plt.xlim(x_start, x_end)
        # plt.tight_layout()

    @classmethod
    def average_dataframe(cls, df, algo_param_list, condition_list=[], min_count=1):
        new_lines = []
        for params in algo_param_list:
            algo_lines = ResultDF.get_dataframe_lines_matching_dict(df=df, val_dict=params)
            if params['algo_name'] == 'PCNN' or params['algo_name'] == 'NoamLSH':
                probe_sizes = set(algo_lines['probe_size'])
                for probe_size in probe_sizes:
                    matching_lines = algo_lines[algo_lines['probe_size'] == probe_size]
                    for condition in condition_list:
                        matching_lines = matching_lines[condition(matching_lines)]
                    if len(matching_lines) == 0:
                        continue
                    avg_ndis = np.average(matching_lines['ndis'])
                    avg_recall = np.average(matching_lines['recall'])
                    count = len(matching_lines)
                    if count < min_count:
                        raise Exception('not enough samples')
                    new_lines.append(
                        {**params, 'probe_size': probe_size, 'ndis': avg_ndis, 'recall': avg_recall, 'count': count})
            elif params['algo_name'] == 'FaissLSH':
                num_flips_set = set(algo_lines['num_flips'])
                for num_flips in num_flips_set:
                    matching_lines = algo_lines[algo_lines['num_flips'] == num_flips]
                    avg_ndis = np.average(matching_lines['ndis'])
                    avg_recall = np.average(matching_lines['recall'])
                    count = len(matching_lines)
                    if count < min_count and count > 0:
                        raise Exception('not enough samples')
                    new_lines.append(
                        {**params, 'num_flips': num_flips, 'ndis': avg_ndis, 'recall': avg_recall, 'count': count})
                # assert count >= min_count
        return pd.DataFrame(new_lines)

    @classmethod
    def ndis_std_dataframe(cls, df, algo_param_list, condition_list=[], min_count=1):
        new_lines = []
        for params in algo_param_list:
            algo_lines = ResultDF.get_dataframe_lines_matching_dict(df=df, val_dict=params)
            if params['algo_name'] == 'PCNN' or params['algo_name'] == 'NoamLSH':
                probe_sizes = set(algo_lines['probe_size'])
                for probe_size in probe_sizes:
                    matching_lines = algo_lines[algo_lines['probe_size'] == probe_size]
                    for condition in condition_list:
                        matching_lines = matching_lines[condition(matching_lines)]
                    if len(matching_lines) == 0:
                        continue
                    ndis_std = np.std(matching_lines['ndis'])
                    # avg_ndis = np.average(matching_lines['ndis'])
                    # avg_recall = np.average(matching_lines['recall'])
                    count = len(matching_lines)
                    if count < min_count:
                        raise Exception('not enough samples')
                    new_lines.append(
                        {**params, 'probe_size': probe_size, 'ndis_std': ndis_std, 'count': count})
            # elif params['algo_name'] == 'FaissLSH':
            #     num_flips_set = set(algo_lines['num_flips'])
            #     for num_flips in num_flips_set:
            #         matching_lines = algo_lines[algo_lines['num_flips'] == num_flips]
            #         avg_ndis = np.average(matching_lines['ndis'])
            #         avg_recall = np.average(matching_lines['recall'])
            #         count = len(matching_lines)
            #         if count < min_count and count > 0:
            #             raise Exception('not enough samples')
            #         new_lines.append(
            #             {**params, 'num_flips': num_flips, 'ndis': avg_ndis, 'recall': avg_recall, 'count': count})
            # assert count >= min_count
        return pd.DataFrame(new_lines)

    @classmethod
    def get_common_variables(cls, algo_param_list):

        first_algo = algo_param_list[0]
        common_params = first_algo.copy()
        del common_params['algo_name']
        diff_params = set()

        for algo_param in algo_param_list:
            pruned_common_params = common_params.copy()
            for key in common_params:
                if key not in algo_param or algo_param[key] != common_params[key]:
                    del pruned_common_params[key]
                    diff_params.add(key)
            common_params = pruned_common_params
            for key in algo_param:
                if key not in common_params:
                    diff_params.add(key)

        return common_params

    @classmethod
    def beautify_params(cls, params):
        pretty_keys = {
            'algo_name': 'algo_name',
            'size_nn': 'size_nn',
            'embedding_class': 'emb_class',
            'num_bits': 'nbit',
            'num_tables': 'ntable',
            'code_dim': 'cdim',
            'prrot_iters': 'emb_iters',
            'seed': 'seed',
            'alpha': '$\\alpha$'

        }
        pretty_names = {
            'NoamLSH': 'LSH',
            'IVF': 'IVF',
            'FaissLSH': 'FaissLSH',
            'PCNN': 'PCNN'
        }
        new_params = {}
        for key, value in params.items():
            new_value = value if value not in pretty_names else pretty_names[value]
            new_key = key if key not in pretty_keys else pretty_keys[key]
            new_params[new_key] = new_value
        return new_params

    @classmethod
    def plot_ndis_std(cls, *,
                      df,
                      algo_param_list):
        bars = []
        bar_width = 0.25
        common_variables = cls.get_common_variables(algo_param_list)
        for i, algo_params in enumerate(algo_param_list):
            val_map = {}
            matching_lines = ResultDF.get_dataframe_lines_matching_dict(df=df, val_dict=algo_params)
            if len(matching_lines) == 0:
                continue
            x_vals = np.log2(matching_lines['probe_size'])
            y_vals = matching_lines['ndis_std']

            # get label for algo
            reduced_params = algo_params.copy()
            [reduced_params.pop(x) for x in common_variables]
            pretty_params = cls.beautify_params(reduced_params)
            plt.bar(x_vals + i * bar_width, y_vals, width=bar_width, label=Algo.name_from_params(pretty_params))
        # plt.yscale('log')
        plt.legend(loc='best', prop={'size': 12})

    @classmethod
    def pareto_frontier_dataframe(cls, df, algo_param_list):
        new_lines = []

        for params in algo_param_list:
            algo_lines = ResultDF.get_dataframe_lines_matching_dict(df=df, val_dict=params)

            # get convex hull
            points = algo_lines[['recall', 'ndis']].to_numpy()

            if len(points) == 0:
                continue
            ch = scipy.spatial.ConvexHull(points)

            hull_indices = ch.vertices

            hull_lines = algo_lines.iloc[hull_indices]

            pareto_frontier = []
            for _, cur_line in hull_lines.iterrows():
                to_add = True
                for line in pareto_frontier:
                    if cls._dominates(line, cur_line):
                        to_add = False
                        break

                if to_add:
                    new_frontier = [cur_line]
                    for line in pareto_frontier:
                        if not cls._dominates(cur_line, line):
                            new_frontier.append(line)
                    pareto_frontier = new_frontier

            new_lines.extend(pareto_frontier)
        return pd.DataFrame(new_lines)

    @classmethod
    def _dominates(cls, l1, l2):
        if l1['recall'] > l2['recall'] and l1['ndis'] < l2['ndis']:
            return True
        return False
