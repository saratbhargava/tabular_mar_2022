import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", nargs="+")
    value = parser.parse_args()

    print(value.model_filename)
