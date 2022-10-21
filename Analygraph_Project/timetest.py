import time
import main_python
import main

# start_time = time.perf_counter()
# main.main()
# end_time = time.perf_counter()


start_copy = time.perf_counter()
main_python.main()
end_copy = time.perf_counter()

# print("Cython performance:" + str(end_time-start_time))
print("Python performance:" + str(end_copy-start_copy))

