import openml

benchmarks = openml.study.get_suite("OpenML-CC18")

# for x in benchmarks.tasks:
#     print(x)


task = openml.tasks.get_task(1501)
xs,ys  = task.get_X_and_y()
print(xs,ys)