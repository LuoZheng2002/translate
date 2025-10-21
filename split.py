



def split_odd_even_lines(input_file, odd_output_file, even_output_file):
    with open(input_file, 'r') as infile, \
         open(odd_output_file, 'w') as oddfile, \
         open(even_output_file, 'w') as evenfile:

        for i, line in enumerate(infile, start=1):
            if i % 2 == 1:
                oddfile.write(line)
            else:
                evenfile.write(line)


# Example usage:
split_odd_even_lines('chinese_auto_revised.json', 'chinese_questions.json', 'chinese_answers.json')