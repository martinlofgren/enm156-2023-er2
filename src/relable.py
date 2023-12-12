import os
path = "" #path to folder

if __name__ == '__main__':
    for file in os.listdir(path):
        if "synthetic" in file:
            data = None
            with open(path + file, 'r') as openfile:
                data = openfile.read()
                data = data.split('\n')
                for i, line in enumerate(data):                 
                    data[i] = '0' + line[1:]
                data = '\n'.join(data)
            with open(path + file, 'w') as openfile:
                openfile.write(data)