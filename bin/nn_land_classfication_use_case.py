import tensorflow as tf

import src.execution.main_executor


def main():
    executor = src.execution.main_executor.MainExecutor(display=True)

    executor.execute_land_use_classification_use_case(
        network_name="03.02.2022_local//network_D_1903240222",
        sat_img_list=[
            "elblag.jpg"
            "france.jpg",
            "usa_florida.jpg",
            "mexico.jpg"
        ]
    )

if __name__ == "__main__" :
        main()
