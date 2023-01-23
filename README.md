# rodes-sorter-genetic-algorithm

Project for Genetic Algorithms course.

## How to run

You will need Python >= 3.8

NOTE: `py` is python launcher which comes by default when installing Python
on a machine. If you do not have that launcher, 
use `python` (or `python3`).

1. Create virtual environment in preferred IDE or in console using:
    ```commandline
    py -m venv venv
    ```
1. Activate virtual environment:
    - for Windows users:
      ```commandline
      venv\Scripts\activate
      ``` 
    - for Linux users:
      ```commandline
      source venv/bin/activate
      ``` 
1. Install all required modules:
    ```commandline
    pip install requirements.txt
    ``` 
1. After installation, you can run program:
   ```commandline
   py src/main.py
   ```

Program by default looks for input file `prety.txt` in the project's root directory.
If you want to use other input file, modify the `filepath` variable in `src/main.py` file.