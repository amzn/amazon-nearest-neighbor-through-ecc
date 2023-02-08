# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools

import sys
import os

os.chdir('/home/ec2-user/PycharmProjects/PolarANNS')
sys.path.append('/home/ec2-user/PycharmProjects/PolarANNS')
# if sys.path[0] == '/home/ec2-user/PycharmProjects/PolarANNS/src':
#     sys.path[0] = '/home/ec2-user/PycharmProjects/PolarANNS'
# print(sys.path)

from src.algos.polar_codes import PolarCodec, ExhaustiveCodec
from src.algos.utils.polar_utils import PolarUtils
from src.testing import *
from src.plotting import *
from src.algos import *
from src.algos.factories import DataModelFactory, IndexFactory, MaskFactory


def test_algos(datamodel,
               dm_filename,
               num_bits_list=[28],
               num_tables_list=[1],
               code_dim_list=[512],
               num_flips_list=list(range(6)),
               probe_size_list=[2 ** i for i in range(10)],
               seed_list=[1264],
               embedding_class_list=['hp'],
               do_pcnn=False,
               do_faiss_lsh=False,
               do_lsh=False,
               do_faiss_ivf=False,
               volatile=True,
               ephemeral=True
               ):
    print('test!')
    # data params
    dim = datamodel.dim
    size_nn_list = [1, 5, 10, 50, 100]
    num_queries = 5000

    lsh_args_dict = {
        'num_bits': num_bits_list,
        'num_tables': num_tables_list,
        'seed': seed_list,
        'embedding_class': embedding_class_list
    }

    pcnn_args_dict = dict(lsh_args_dict)
    pcnn_args_dict['code_dim'] = code_dim_list
    # pcnn_args_dict['embedding_class'] = ['shp']
    # pcnn_args_dict['prrot_iters'] = [2]

    # ivf params
    n_list_list = [4 ** i for i in range(6, 8)]
    n_probe_list = [2 ** i for i in range(5)]
    ivf_args_dict = {
        'n_list': n_list_list
    }

    pcnn_configs = [dict(zip(pcnn_args_dict, x)) for x in itertools.product(*pcnn_args_dict.values())]
    lsh_configs = [dict(zip(lsh_args_dict, x)) for x in itertools.product(*lsh_args_dict.values())]
    ivf_configs = [dict(zip(ivf_args_dict, x)) for x in itertools.product(*ivf_args_dict.values())]

    # ibivf params
    n_list_list = [4 ** i for i in range(5, 8)]
    n_probe = 1

    if do_faiss_lsh:
        for config in lsh_configs:
            faiss_lsh = FaissLSH(dim=dim,
                                 **config)

            index_config = faiss_lsh.get_index_config()
            faiss_lsh_index = IndexFactory.get_index(faiss_lsh.INDEX_CLASS, datamodel=datamodel, ephemeral=ephemeral,
                                                     **index_config)
            faiss_lsh.set_index(faiss_lsh_index)

            Testing.calc_algo_recall(faiss_lsh,
                                     datamodel=datamodel,
                                     dm_filename=dm_filename,
                                     num_queries=num_queries,
                                     size_nn_list=size_nn_list,
                                     change_func=faiss_lsh.set_num_flips,
                                     change_param_name='num-flips',
                                     change_param_values=num_flips_list,
                                     # overwrite=overwrites['FaissLSH'],
                                     volatile=volatile,
                                     )

    if do_faiss_ivf:
        for config in ivf_configs:
            ivf = FaissIVF(dim=dim, **config)
            index_config = ivf.get_index_config()
            ivf_index = IndexFactory.get_index(ivf.INDEX_CLASS, datamodel=datamodel, ephemeral=ephemeral,
                                               **index_config)
            ivf.set_index(ivf_index)

            Testing.calc_algo_recall(ivf,
                                     datamodel=datamodel,
                                     dm_filename=dm_filename,
                                     num_queries=num_queries,
                                     size_nn_list=size_nn_list,
                                     # approx_factor=approx_factor,
                                     change_func=ivf.set_n_probe,
                                     change_param_name='nprobe',
                                     change_param_values=n_probe_list,
                                     # overwrite=overwrites['FaissIVF'],
                                     volatile=volatile,
                                     )

    if do_lsh:
        for config in lsh_configs:
            lsh = LSH(dim=dim,
                      **config)

            index_config = lsh.get_index_config()
            lsh_index = IndexFactory.get_index(LSHIndex, datamodel=datamodel, ephemeral=ephemeral, **index_config)
            lsh.set_index(lsh_index)

            Testing.calc_algo_recall(lsh,
                                     datamodel=datamodel,
                                     dm_filename=dm_filename,
                                     num_queries=num_queries,
                                     size_nn_list=size_nn_list,
                                     # approx_factor=approx_factor,
                                     change_func=lsh.set_probe_size,
                                     change_param_name='probe-size',
                                     change_param_values=probe_size_list,
                                     # overwrite=overwrites['LSH'],
                                     volatile=volatile,
                                     )

    if do_pcnn:
        for config in pcnn_configs:
            code_mask, _ = MaskFactory.get_mask(N=config['code_dim'], K=config['num_bits'])
            mask_val = code_mask.mask
            pcnn = PCNN(dim=dim,
                        code_mask_name='bsc',
                        code_mask=mask_val,
                        **config)

            index_config = pcnn.get_index_config()
            pcnn_index = IndexFactory.get_index(PCNNIndex, datamodel=datamodel, ephemeral=ephemeral, **index_config)
            pcnn.set_index(pcnn_index)

            Testing.calc_algo_recall(pcnn,
                                     datamodel=datamodel,
                                     dm_filename=dm_filename,
                                     num_queries=num_queries,
                                     size_nn_list=size_nn_list,
                                     change_func=pcnn.set_probe_size,
                                     change_param_name='probe-size',
                                     change_param_values=probe_size_list,
                                     volatile=volatile,
                                     )
    print('done with testing!')


def get_bigann(size_ds=10000000):
    bigann_dm, bigann_dm_filename = DataModelFactory.get_datamodel(BIGANNDataModel, size_ds=size_ds,
                                                                   dim=BIGANNDataModel.BIGANN_DIM)
    # bigann_dm, bigann_dm_filename = BIGANNDataModel.get_datamodel(size_ds=size_ds,
    #                                                               dim=BIGANNDataModel.BIGANN_DIM)
    return bigann_dm, bigann_dm_filename


def get_df_from_dm_filename(dm_filename):
    noext = os.path.splitext(dm_filename)[0]
    num_queries = 5000
    return f'{noext}___numquer={num_queries}.csv'


def build_configs(*, alpha_list=[1.1],
                  size_nn_list=[1],
                  num_bits_list=[28],
                  code_dim_list=[512],
                  num_tables_list=[1],
                  condition_list=[],
                  # use_embedding = True,
                  embedding_class_list=None,
                  seed_list=None,
                  prrot_iters_list=None,
                  do_pcnn=False,
                  do_lsh=False,
                  do_faiss_lsh=False,
                  do_ivf=False,
                  n_list_list=None
                  ):
    algo_param_list = []
    # plot_param_list = []

    if do_pcnn:
        pcnn_args_dict = {
            'num_bits': num_bits_list,
            'num_tables': num_tables_list,
            'code_dim': code_dim_list,
            'alpha': alpha_list,
            'size_nn': size_nn_list,
        }
        if embedding_class_list is not None:
            pcnn_args_dict['embedding_class'] = embedding_class_list
            pcnn_args_dict['prrot_iters'] = prrot_iters_list

        if seed_list is not None:
            pcnn_args_dict['seed'] = seed_list

        # product dict
        pcnn_configs = []
        for x in itertools.product(*pcnn_args_dict.values()):
            config = dict(zip(pcnn_args_dict, x))
            if embedding_class_list is not None:
                if config['embedding_class'] != 'shp' and config['prrot_iters'] is not None:
                    continue
                elif config['embedding_class'] != 'shp':
                    del config['prrot_iters']
                if config['embedding_class'] == 'shp' and config['prrot_iters'] is None:
                    continue

            pcnn_configs.append(config)

        # pcnn_configs = [
        #     dict(zip(pcnn_args_dict, x)) for x in itertools.product(*pcnn_args_dict.values())
        #     if not (dict(zip(pcnn_args_dict, x)))['embedding_class']
        # ]

        # bsc
        for config in pcnn_configs:
            config['algo_name'] = 'PCNN'
            algo_param_list.append(config)
            # plot_param_list.append('probe_size')

    if do_lsh or do_faiss_lsh:
        lsh_args_dict = {
            'num_bits': num_bits_list,
            'num_tables': num_tables_list,
            # 'embedding_class': embedding_class_list,
            # 'seed' : seed_list,
            'alpha': alpha_list,
            'size_nn': size_nn_list
        }
        if seed_list is not None:
            lsh_args_dict['seed'] = seed_list
        if embedding_class_list is not None:
            lsh_args_dict['embedding_class'] = embedding_class_list
            lsh_args_dict['prrot_iters'] = prrot_iters_list

        lsh_configs = []
        for x in itertools.product(*lsh_args_dict.values()):
            config = dict(zip(lsh_args_dict, x))
            if embedding_class_list is not None:
                if config['embedding_class'] != 'shp' and config['prrot_iters'] is not None:
                    # print('hi')
                    continue
                elif config['embedding_class'] != 'shp':
                    del config['prrot_iters']
                if config['embedding_class'] == 'shp' and config['prrot_iters'] is None:
                    continue

            lsh_configs.append(config)

        # lsh_configs = [dict(zip(lsh_args_dict, x)) for x in itertools.product(*lsh_args_dict.values())]

        if do_lsh:
            for config in lsh_configs:
                new_conf = config.copy()
                new_conf['algo_name'] = 'NoamLSH'
                algo_param_list.append(new_conf)
        if do_faiss_lsh:
            for config in lsh_configs:
                new_conf = config.copy()
                new_conf['algo_name'] = 'FaissLSH'
                algo_param_list.append(new_conf)

    if do_ivf:
        ivf_args_dict = {
            'n_list': n_list_list,
            'alpha': alpha_list,
            'size_nn': size_nn_list
        }
        ivf_configs = [dict(zip(ivf_args_dict, x)) for x in itertools.product(*ivf_args_dict.values())]

        for config in ivf_configs:
            config['algo_name'] = 'FaissIVF'
            algo_param_list.append(config)

    return algo_param_list


def plot_average_recall(dm_params_list,
                        legend=True,
                        color=None,
                        pretty=False,
                        title="missing title",
                        use_dashes=True,
                        condition_list=[],
                        use_markers=False,
                        x_start=None,
                        x_end=None,
                        min_count=1,
                        **kwargs):
    df_filenames = []
    for dm_params in dm_params_list:
        dm_filename = DataModelFactory.get_dm_filename_from_params(dm_params)
        df_filenames.append(get_df_from_dm_filename(dm_filename))

    dfs = []
    for df_filename in df_filenames:
        dfs.append(ResultDF.read_dataframe(df_filename=df_filename))
    merged_df = pd.concat(dfs)
    algo_param_list = build_configs(**kwargs)

    avg_df = Plotting.average_dataframe(merged_df, algo_param_list, condition_list=condition_list, min_count=min_count)
    Plotting.plot_recall_ndis(algo_param_list=algo_param_list,
                              df=avg_df,
                              legend=legend,
                              color=color,
                              pretty=pretty,
                              condition_list=condition_list,
                              use_dashes=use_dashes,
                              use_markers=use_markers,
                              x_start=x_start,
                              x_end=x_end,
                              title=f'{title}')


def draw_std(dm_params_list,
             min_count=1,
             **kwargs):
    df_filenames = []
    for dm_params in dm_params_list:
        dm_filename = DataModelFactory.get_dm_filename_from_params(dm_params)
        df_filenames.append(get_df_from_dm_filename(dm_filename))

    dfs = []
    for df_filename in df_filenames:
        dfs.append(ResultDF.read_dataframe(df_filename=df_filename))
    merged_df = pd.concat(dfs)

    algo_param_list = build_configs(**kwargs)

    # avg_df = Plotting.average_dataframe(merged_df, algo_param_list, condition_list = condition_list, min_count=min_count)
    var_df = Plotting.ndis_std_dataframe(merged_df, algo_param_list, min_count=min_count)
    Plotting.plot_ndis_std(algo_param_list=algo_param_list,
                           df=var_df)


def plot_frontier_recall(dm_params_list,
                         legend=True,
                         color=None,
                         pretty=False,
                         title="missing title",
                         use_dashes=True,
                         use_markers=True,
                         **kwargs):
    df_filenames = []
    for dm_params in dm_params_list:
        dm_filename = DataModelFactory.get_dm_filename_from_params(dm_params)
        df_filenames.append(get_df_from_dm_filename(dm_filename))

    dfs = []
    for df_filename in df_filenames:
        dfs.append(ResultDF.read_dataframe(df_filename=df_filename))
    merged_df = pd.concat(dfs)

    algo_param_list = build_configs(**kwargs)

    pf_df = Plotting.pareto_frontier_dataframe(merged_df, algo_param_list)
    Plotting.plot_recall_ndis(algo_param_list=algo_param_list,
                              df=pf_df,
                              legend=legend,
                              color=color,
                              pretty=pretty,
                              use_dashes=use_dashes,
                              use_markers=use_markers,
                              title=f'{title}')


def test_speedy_hp_embedding():
    yandex_dm, yandex_filename = get_yandexdeep(200000)
    hp = SpeedyHyperplaneEmbedding(source_dim=96, dest_dim=128, seed=1234)
    hp.generate()
    words = hp.apply(yandex_dm.dataset)
    print(words.shape)


def test_polar_code():
    """
    This function tests the recall of the PolarCodec for recovering the closest codewords using various list sizes.
    It supports our choice of the list size as a function of the desired number of nearest codewords (e.g., list of 16 for single nearest neighbor, and so on).
    :return:
    """
    dim = 64
    data_dim = 16
    pc_list = 16
    num_best = 1

    size_ds = 300
    mask_obj, _ = MaskFactory.get_mask(N=dim, K=data_dim)
    mask = mask_obj.mask
    my_list_codec = PolarCodec(N=dim, K=data_dim, L=pc_list, mask=mask)
    list_codec = ExhaustiveCodec(N=dim, K=data_dim, L=num_best, mask=mask)
    dm, _ = DataModelFactory.get_datamodel(UniformDataModel, size_ds=size_ds, dim=dim)
    # dm, _ = UniformDataModel.get_datamodel(size_ds=size_ds, dim=dim)
    points = dm.get_dataset()
    my_list_distances = np.zeros(shape=points.shape[:-1] + (num_best,), dtype=int)
    list_distances = np.zeros(shape=points.shape[:-1] + (num_best,), dtype=int)

    my_mat_list_results = my_list_codec.simple_decode(points, to_info=False)[..., :num_best, :]

    for pind in range(len(points)):
        point = points[pind]
        unpacked = unpack_point(point, dim)
        llrs = PolarUtils.get_llrs(unpacked)

        # my_mat_list_res = my_list_codec.decode(llrs, to_info=False)[...,:num_best, :]
        # packed_mat_list_res = pack_point(my_mat_list_res, dim)
        packed_mat_list_res = my_mat_list_results[pind]

        list_res = list_codec.decode(llrs, to_info=False)
        packed_list_res = pack_point(list_res, dim)

        # get distances
        current_distances = get_binary_dist(packed_list_res, point)
        my_current_distances = get_binary_dist(packed_mat_list_res, point)
        my_current_distances.sort()
        current_distances.sort()
        my_list_distances[pind, :] = my_current_distances
        list_distances[pind, :] = current_distances

    # calculate recall of method
    dist_thres = np.max(list_distances, axis=-1)
    soft_counts = np.count_nonzero(list_distances == dist_thres[..., np.newaxis], axis=-1)
    equal_counts = np.count_nonzero(my_list_distances == dist_thres[..., np.newaxis], axis=-1)
    equal_rec = np.minimum(equal_counts, soft_counts)
    smaller_rec = np.count_nonzero(my_list_distances < dist_thres[..., np.newaxis], axis=-1)
    recall = (equal_rec + smaller_rec) / num_best
    print(recall)


def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time at start of main = ", current_time)

    size_ds = 10000000
    # bigann_dm, bigann_dm_filename = DataModelFactory.get_datamodel(BIGANNDataModel, size_ds=size_ds,
    #                                                                dim=BIGANNDataModel.BIGANN_DIM)
    ydeep_dm, ydeep_dm_filename = DataModelFactory.get_datamodel(YandexDeepDataModel, size_ds=size_ds,
                                                                 dim=YandexDeepDataModel.YANDEX_DEEP_DIM)
    # ytti_dm, ytti_dm_filename = DataModelFactory.get_datamodel(YandexTTIDataModel, size_ds=size_ds,
    #                                                            dim=YandexTTIDataModel.YANDEX_TTI_DIM)

    test_algos(ydeep_dm,
               ydeep_dm_filename,
               num_bits_list=[28],
               num_tables_list=[1],
               code_dim_list=[32, 64, 128, 256, 512, 1024],
               probe_size_list=[2 ** i for i in range(14)],
               seed_list=[42],
               embedding_class_list=['ssrae'],
               do_pcnn=True,
               do_lsh=False,
               volatile=True,
               ephemeral=True)

    print('done!')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time at dim of main = ", current_time)



if __name__ == '__main__':
    main()

