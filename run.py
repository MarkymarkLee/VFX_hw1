from main import run_process


def main():
    data_folder = 'data/mine'

    all_hdr_methods = ['paul', 'robertson', 'nayar']

    for hdr_method in all_hdr_methods:
        print(
            f"Running HDR method: {hdr_method}, with all tonemap methods...")

        result_directory = f"results/{hdr_method}"

        gamma = 1 if hdr_method == "nayar" else 2.2

        run_process(data_folder, hdr_method, "all",
                    align=True, result_directory=result_directory, gamma=gamma)


if __name__ == "__main__":
    main()
