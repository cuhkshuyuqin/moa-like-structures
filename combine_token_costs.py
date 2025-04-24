FILE_1 = "results/alpaca/2025-04-22-21-09-37_token_costs.txt"
FILE_2 = "results/alpaca/2025-04-23-12-29-13_token_costs.txt"
FILE_combined = "results/alpaca/2025-04-22-21-09-37_token_costs_combined.txt"

with open(FILE_1, "r") as file_1:
    with open(FILE_2, "r") as file_2:
        with open(FILE_combined, "w") as file_combined:
            file_1_contents = file_1.readlines()
            file_1_numbers = []
            for line in file_1_contents:
                split_line = line.split(":")
                for split in split_line:
                    if split.strip().isdigit():
                        file_1_numbers.append(int(split.strip()))

            file_2_contents = file_2.readlines()
            file_2_numbers = []
            for line in file_2_contents:
                split_line = line.split(":")
                for split in split_line:
                    if split.strip().isdigit():
                        file_2_numbers.append(int(split.strip()))

            assert len(file_1_numbers) == len(file_2_numbers)

            numbers_combined = []
            for i in range(len(file_1_numbers)):
                numbers_combined.append(file_1_numbers[i] + file_2_numbers[i])
            
            number_counter = 0
            for line in file_1_contents:
                line_combined = ""

                split_line = line.split(":")
                number_contained = False
                for split in split_line:
                    if split.strip().isdigit():
                        number_contained = True
                
                if number_contained:
                    line_combined = split_line[0] + ": " + str(numbers_combined[number_counter]) + "\n"
                    number_counter += 1
                else:
                    line_combined = line
                
                file_combined.write(line_combined)
