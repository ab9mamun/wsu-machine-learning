We used Python 3.7.4 to run the code.
Download and Extract the zip file.
Create a data folder in the extracted directory and place the data files in the data folder. We will need the two following files inside the "data" folder:

fashion-mnist_train.csv
fashion-mnist_test.csv

If you are on a Linux machine, you can do the following to run the code

chmod +x script.sh
./script.sh

The script.sh file uses the python3 command to run the q_learning.py and cnn.py files.

Note that the output of the python files are written to the q_learning-output.txt file and cnn-output.txt file respectively, and the program does not print that much information on the terminal.

If you have a GUI, then you can see the graph opened in different windows. The program will also generate a png file and save it in in the directory from which the code is running.

So, if you are on a Linux server, you can transfer those png files to a workstation (Desktop/laptop) and watch them there.

The following python libraries are required.
pandas
numpy
sklearn
matplotlib
tensorflow


If you do not have one of them, please install the library with pip3 and try to run the code again.
