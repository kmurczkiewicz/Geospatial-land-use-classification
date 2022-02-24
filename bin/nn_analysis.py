import src.execution.main_executor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    executor = src.execution.main_executor.MainExecutor(display=True)

    executor.execute_analyze_networks()

#     executor.execute_analyze_single_network(
#         network_name="network_D_2142310122",
#         layer_num=1,
#         image_path="River/River_119.jpg"
#     )

if __name__ == "__main__" :
        main()
