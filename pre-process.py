def remove_non_ascii(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        ascii_content = ''.join(char for char in content if char.isascii())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ascii_content)

input_file = 'your file path'
output_file = 'your file path'
remove_non_ascii(input_file, output_file)
