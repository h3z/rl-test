'''
用来生成配置赛题的yaml文件
主要的思路为：从example处得到动态参数，加上静态参数，合并为
main.yml的整体yaml文件。
运行main.py时，从utilize/main.yml文件处读取信息
'''
import os
import re
import yaml
import pandas as pd
import example


def _round(x):
    return round(x, 2)


def _get_ld_info(ld_name, adjld_name, stoenergy_name):
    ld_type = [0] * len(ld_name)
    adjld_ids = [i for i, x in enumerate(ld_name) if x in adjld_name]
    stoenergy_ids = [i for i, x in enumerate(ld_name) if x in stoenergy_name]

    for i in adjld_ids:
        ld_type[i] = 1
    for j in stoenergy_ids:
        ld_type[j] = 2
    return adjld_ids, stoenergy_ids, ld_type


def read_grid_data():
    grid = example.Print()
    dataset_path = args.dataset_path

    ld_p_filepath = os.path.join(dataset_path, 'load_p.csv')
    ld_q_filepath = os.path.join(dataset_path, 'load_q.csv')
    gen_p_filepath = os.path.join(dataset_path, 'gen_p.csv')
    gen_q_filepath = os.path.join(dataset_path, 'gen_q.csv')
    max_renewable_gen_p_filepath = os.path.join(dataset_path, 'max_renewable_gen_p.csv')
    keep_decimal_digits = 2
    assert os.path.isfile(ld_p_filepath), "Cannot find the data file."
    assert os.path.isfile(gen_p_filepath), "Cannot find the data file."

    grid.readdata(1, ld_p_filepath, ld_q_filepath, gen_p_filepath, gen_q_filepath)
    grid.env_feedback(grid.name_unp, grid.itime_unp[0], [], 1, [], [0] * 15)

    # 机组个数
    gen_num = len(grid.unname)

    # 不同类型机组机组编号
    gen_type = grid.gen_type
    thermal_ids = [i for i, x in enumerate(grid.gen_type) if x == 1]
    renewable_ids = [i for i, x in enumerate(grid.gen_type) if x == 5]
    balanced_ids = [i for i, x in enumerate(grid.gen_type) if x == 2]
    balanced_id = balanced_ids[0]

    # 机组有功出力上下限
    gen_p_max = [_round(x) for x in grid.gen_plimit]
    gen_p_min = [_round(x) for x in grid.gen_pmin]

    # 机组有功出力上下限
    gen_q_max = [_round(x) for x in grid.gen_qmax]
    gen_q_min = [_round(x) for x in grid.gen_qmin]

    # 机组电压上下限
    gen_v_max = [_round(x) for x in grid.gen_vmax]
    gen_v_min = [_round(x) for x in grid.gen_vmin]

    # 节点名称
    bus_name = grid.busname

    # 节点电压上下限
    bus_v_max = [_round(x) for x in grid.bus_vmax]
    bus_v_min = [_round(x) for x in grid.bus_vmin]

    # 线路名称
    ln_name = grid.lnname
    # 线路个数
    ln_num = len(grid.lnname)
    # 线路电流热极限
    ln_thermal_limit = [_round(x) for x in grid.line_thermal_limit]

    # 断面个数
    sample_num = pd.read_csv(ld_p_filepath).shape[0]

    un_nameindex_key = list(grid.un_nameindex.keys())
    un_nameindex_value = list(grid.un_nameindex.values())

    # 负荷名称
    ld_name = grid.ldname
    # 可调负荷名称
    adjld_name = grid.adjld_name
    # 储能名称
    stoenergy_name = grid.energystorage_name
    # 不同类型负荷编号（普通负荷、可调负荷、储能设备）
    adjld_ids, stoenergy_ids, ld_type = _get_ld_info(ld_name, adjld_name, stoenergy_name)

    # 可调负荷个数
    adjld_num = len(grid.adjld_name)
    # 可调负荷容量
    adjld_capacity = grid.adjld_capacity
    # # 可调负荷下限
    # adjld_min = [-i for i in grid.adjld_capacity]
    # 可调负荷上调节速率
    adjld_uprate = grid.adjld_uprate
    # 可调负荷下调节速率
    adjld_dnrate = grid.adjld_dnrate

    # 储能个数
    stoenergy_num = len(grid.energystorage_name)
    # 储能容量
    stoenergy_capacity = grid.energystorage_capacity
    # 储能充电上限
    stoenergy_chargerate_max = grid.energystorage_chargerate
    # 储能放电上限
    stoenergy_dischargerate_max = grid.energystorage_dischargerate

    dict_ = {
        'gen_num': gen_num,
        'gen_type': gen_type,
        'gen_p_max': gen_p_max,
        'gen_p_min': gen_p_min,
        'gen_q_max': gen_q_max,
        'gen_q_min': gen_q_min,
        'gen_v_max': gen_v_max,
        'gen_v_min': gen_v_min,
        'renewable_ids': renewable_ids,
        'thermal_ids': thermal_ids,
        'balanced_id': balanced_id,
        'gen_p_filepath': gen_p_filepath,
        'gen_q_filepath': gen_q_filepath,
        'max_renewable_gen_p_filepath': max_renewable_gen_p_filepath,

        'ln_name': ln_name,
        'ln_num': ln_num,
        'ln_thermal_limit': ln_thermal_limit,

        'ld_p_filepath': ld_p_filepath,
        'ld_q_filepath': ld_q_filepath,
        'ld_name': ld_name,
        'adjld_num': adjld_num,
        'adjld_name': adjld_name,
        'adjld_capacity': adjld_capacity,
        'adjld_uprate': adjld_uprate,
        'adjld_dnrate': adjld_dnrate,
        'stoenergy_num': stoenergy_num,
        'stoenergy_name': stoenergy_name,
        'stoenergy_capacity': stoenergy_capacity,
        'stoenergy_chargerate_max': stoenergy_chargerate_max,
        'stoenergy_dischargerate_max': stoenergy_dischargerate_max,

        'bus_name': bus_name,
        'bus_v_max': bus_v_max,
        'bus_v_min': bus_v_min,

        'sample_num': sample_num,

        'un_nameindex_key': un_nameindex_key,
        'un_nameindex_value': un_nameindex_value,

        'adjld_ids': adjld_ids,
        'stoenergy_ids': stoenergy_ids,
        'ld_type': ld_type
    }
    return dict_


# 需要读取grid信息才能得到的yaml称为dynamic
def create_dynamic_yml():
    dict_ = read_grid_data()
    with open('utilize/parameters/dynamic.yml', 'w+', encoding='utf-8') as f:
        for key, val in dict_.items():
            stream = yaml.dump({key: val}, default_flow_style=True)
            f.write(re.sub(r'{|}', '', stream))


def merge_dynamic_static_yml():
    with open('utilize/parameters/dynamic.yml', 'r', encoding='utf-8') as f:
        dict_dynamic = yaml.load(f, Loader=yaml.Loader)
    with open('utilize/parameters/static.yml', 'r', encoding='utf-8') as f:
        dict_static = yaml.load(f, Loader=yaml.Loader)

    dict_static.update(dict_dynamic)
    with open('utilize/parameters/main.yml', 'w+', encoding='utf-8') as f:
        for key, val in dict_static.items():
            stream = yaml.dump({key: val}, default_flow_style=True)
            f.write(re.sub(r'{|}', '', stream))


def main():
    create_dynamic_yml()
    merge_dynamic_static_yml()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # python yml_creator.py - -dataset_path '/data'

    parser.add_argument('--dataset_path', default="data", type=str, help='The path of the dataset.')
    args = parser.parse_args()

    main()
