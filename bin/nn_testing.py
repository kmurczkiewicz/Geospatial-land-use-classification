import src.execution.main_executor


def main():
    executor = src.execution.main_executor.MainExecutor(display=False)
    data_dict = executor.stage_prepare_data(read_head=False)
    data = executor.stage_load_data(data_dict)
    executor.stage_test_saved_networks(data, [], False)

if __name__ == "__main__" :
        main()

