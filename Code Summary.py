import os
import openai

openai.api_key = 'sk-yours-key'

def analyze_java_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.java'):
            java_file = os.path.join(folder_path, filename)
            output_file = os.path.splitext(java_file)[0] + '.txt'  
            analyze_and_write_summary(java_file, output_file)

def analyze_and_write_summary(java_file, output_file):
    with open(java_file, 'r', encoding='utf-8') as file:
        code = file.read()
        summary = summarize_class(code)
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(summary)

def summarize_class(code):
    text = f"Please tell me the purpose of this Java file by using a verb object phrase, must shorter than 10 words ：:"
    a = text + f"{code}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": a}
        ]
        
)
    summary = response.choices[0].message.content
    return summary


java_folder = 'yours-path'
analyze_java_files_in_folder(java_folder)
