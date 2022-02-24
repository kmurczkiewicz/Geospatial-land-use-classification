import src.execution.main_executor

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_data_analysis()

if __name__ == "__main__" :
        main()

