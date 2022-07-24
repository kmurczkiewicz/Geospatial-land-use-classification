import src.execution.main_executor
import os
import pandas


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    pandas.set_option('display.max_rows', None)

    executor = src.execution.main_executor.MainExecutor(display=True)
    # executor.execute_analyze_networks(nn_dir="D_17.03.2022_opt_act")
    executor.execute_analyze_networks(nn_dir="")
    executor.execute_analyze_single_network(
        network_name="D_10.03.2022_opt_act/network_D_2055100322",
        layer_num=1,
        image_path="River/River_119.jpg"
    )

if __name__ == "__main__" :
        main()
