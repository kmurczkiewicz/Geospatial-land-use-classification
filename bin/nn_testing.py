import src.execution.main_executor

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_test_networks(
        networks_to_test = [
            "network_D_2142310122"
        ]
    )

if __name__ == "__main__" :
        main()
