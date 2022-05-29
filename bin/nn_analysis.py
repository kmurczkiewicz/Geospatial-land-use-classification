import src.execution.main_executor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
pandas.set_option('display.max_rows', None)

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)

    # executor.execute_analyze_networks(nn_dir="10.03.2022_opt_act")

    executor.execute_analyze_networks(nn_dir="")

#     executor.execute_analyze_single_network(
#          network_name="03.02.2022_local/network_D_1903240222",
#          layer_num=1,
#          image_path="River/River_119.jpg"
#     )

if __name__ == "__main__" :
        main()
