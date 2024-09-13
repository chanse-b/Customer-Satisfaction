# Creates a zip file for submission on Gradescope.
# adapted from 224N

import os
import zipfile

required_files = [p for p in os.listdir('.') if p.endswith('.py')] + \
                [p for p in os.listdir('.') if p.endswith('.md')] + \
                [p for p in os.listdir('.') if p.endswith('.png')] + \
                 [f'data/{p}' for p in os.listdir('data')]





def main():
    aid = 'cs129final_project_submission'
    path = os.getcwd()
    with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()
