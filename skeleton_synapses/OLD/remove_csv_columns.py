import csv
from skeleton_synapses.skeleton_utils import CSV_FORMAT

def remove_csv_columns(input_path, output_path, columns_to_remove):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        reader = csv.DictReader(f_in, **CSV_FORMAT)
        output_columns = list(reader.fieldnames)
        for col in columns_to_remove:
            output_columns.remove(col)

        writer = csv.DictWriter(f_out, output_columns, **CSV_FORMAT)
        writer.writeheader()
        for row in reader:
            for col in columns_to_remove:
                del row[col]
            writer.writerow(row)
    

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("columns_to_remove", nargs='+')

    parsed_args = parser.parse_args()
    remove_csv_columns( parsed_args.input_path, 
                        parsed_args.output_path, 
                        parsed_args.columns_to_remove )
