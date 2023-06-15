import subprocess
import time

# Program path
program_path = r'..\..\bin\win64\Release\TabuSearch.exe'

times = []
blocks = [8, 1, 2, 4, 8, 16, 32, 64, 128, 256]
threads = [64, 1, 2, 4, 8, 16, 32, 64, 128, 256]
iterations = [1]

total_c = len(blocks)*len(threads)*len(iterations)
c = 0
print("total:",total_c)
print("blocks,threads,iterations,time,completed")
for b in blocks:
    for t in threads:
        for i in iterations:
            # Construct the command to execute
            command = ['run.bat', program_path, str(b), str(t), str(i), str(0)]

            # Measure execution time
            start_time = time.time()

            # Execute the command
            subprocess.run(command, stdout=subprocess.DEVNULL)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Print the execution time
            times.append(execution_time)
            print(b,t,i,execution_time,str(c+1)+"/"+str(total_c),sep=",")
            c+=1

# print("blocks,threads,iterations,time")
# for b in blocks:
#     for t in threads:
#         for i in iterations:
#             print(b,t,i,times.pop(0))