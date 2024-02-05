import os


input_dir = "../raw_texts/John_Quincy_Adams.txt"
output_dir = "../books/John_Quincy_Adams"

with open(input_dir, 'r') as f:
  text = f.read()

chapters = text.split('CHAPTER ')

os.makedirs(output_dir, exist_ok=True)

os.makedirs(os.path.join(output_dir, 'part1'), exist_ok=True)

# Write each chapter to a separate file
for i, chapter in enumerate(chapters):
  with open(os.path.join(output_dir, 'part1', f'chapter{i}.txt'), 'w') as f:
    f.write(chapter)