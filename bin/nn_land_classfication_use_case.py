import tensorflow as tf
import src.execution.main_executor


def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_land_use_classification_use_case(
        network_name="A_17.03.2022_opt_act/network_A_1428170322",
        sat_img_list=[
            "elblag.jpg",
            "tokio.jpg",
            "bory_tucholskie.jpg"
        ]
    )

if __name__ == "__main__" :
        main()