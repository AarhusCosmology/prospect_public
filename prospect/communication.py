import heapq
import os
import pickle
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Type
from dataclasses import dataclass
from prospect.input import Configuration
from prospect.tasks.base_task import BaseTask
from prospect.tasks.analyse_profile_task import AnalyseProfileTask
from prospect.tasks.initialise import initialise_tasks
from multiprocessing import Condition

@dataclass
class TasksState:
    queued: list[Type[BaseTask]]
    # All task dicts are indexed by the task ids 
    unready: dict[int, Type[BaseTask]]
    ongoing: dict[int, Type[BaseTask]]
    done:    dict[int, Type[BaseTask]]
    failed:  dict[int, Type[BaseTask]]
    dependencies: defaultdict[int, list[int]]

class Scheduler:
    def __init__(self, config: Configuration, state: TasksState = None) -> None:
        if state:
            # Resume from snapshot
            self.tasks = state
            # Required to restart all tasks that were ongoing
            for task in self.tasks.ongoing.values():
                self.push_task(task)
            self.tasks.ongoing = {}
        else:
            # Start from config file
            self.tasks = TasksState([], {}, {}, {}, {}, defaultdict(list))
            new_tasks = initialise_tasks(config)
            for task in new_tasks:
                self.push_task(task)
        self.config = config
        self.condition = Condition()
        self.start_time = time.time()
        self.start_date = datetime.now()
        self.next_dump_time = time.time() + config.io.snapshot_interval
        self.sleep_iteration = 0

    def delegate(self, executor) -> None:
        # Executor is a context managed instance of either mpi4py.futures.MPICommExecutor 
        # or concurrent.futures.ThreadPoolExecutor
        while True:
            if not self.tasks.queued:
                if not self.tasks.ongoing:
                    if not self.tasks.unready:
                        break
                    elif self.tasks.failed:
                        print(f"Exiting PROSPECT with one or more failed tasks.")
                        break

                with self.condition:
                    self.condition.wait()
                continue

            ready_task = self.pop_task()
            task_requirements = [self.tasks.done[idx] for idx in ready_task.required_task_ids]
            executor.submit(ready_task.run_return_self, task_requirements).add_done_callback(self.finalize_task)

    def finalize(self, executor) -> None:
        # Things to do before shutting down
        if self.config.io.write:
            if self.config.run.jobtype == 'profile':
                analysis_task = self.get_profile_analysis()
                analysis_task.run([task for task in self.tasks.done.values() if task.id in analysis_task.required_task_ids])
            self.dump_snapshot()
            self.status_update()

    def push_task(self, task: Type[BaseTask]) -> None:
        # Pushes task and sets dependencies
        task.time_emit = datetime.now()
        if self.is_task_ready(task):
            # Works since BaseTasks are ordered by their priority
            heapq.heappush(self.tasks.queued, task)
        else:
            self.tasks.unready[task.id] = task
        for req in task.required_task_ids:
            self.tasks.dependencies[req].append(task.id)

    def pop_task(self) -> Type[BaseTask]:
        # Pops first task from task queue and moves to ongoing tasks
        ready_task = heapq.heappop(self.tasks.queued)
        ready_task.time_launch = datetime.now()
        self.tasks.ongoing[ready_task.id] = ready_task
        return ready_task

    def is_task_ready(self, task: Type[BaseTask]) -> bool:
        for req in task.required_task_ids:
            if req not in self.tasks.done:
                return False 
        return True
    
    def fail_task(self, task: Type[BaseTask]) -> None:
        print("\n-------------------------------------------------------------------")
        print(f"WARNING! TASK {task.id} RAISED THE FOLLOWING EXCEPTION:\n")
        print(task.error)
        print(f"Moving the failed task to TasksState.failed and continuing.")
        print("-------------------------------------------------------------------\n")
        if hasattr(task, "mcmc.kernel"):
            del task.mcmc.kernel
        del self.tasks.ongoing[task.id]
        self.tasks.failed[task.id] = task
        with self.condition:
            self.condition.notify_all()

    def finalize_task(self, future) -> None:
        finished_task = future.result() # BaseTask.run_return_self returns the task that was run 

        # Handle tasks that raised an exception
        if not finished_task.success:
            self.fail_task(finished_task)
            return

        # Continue, assuming task finished succesfully
        finished_task.time_finish = datetime.now()
        del self.tasks.ongoing[finished_task.id]
        self.tasks.done[finished_task.id] = finished_task

        # Add tasks emitted by finished_task
        for new_task in finished_task.emit_tasks():
            new_task.emitted_by = finished_task.id
            self.push_task(new_task)

        # Update dependencies
        for dependency_id in self.tasks.dependencies[finished_task.id]:
            if dependency_id in self.tasks.unready:
                # Check if dependency is now ready
                if self.is_task_ready(self.tasks.unready[dependency_id]):
                    new_task = self.tasks.unready.pop(dependency_id)
                    heapq.heappush(self.tasks.queued, new_task)

        with self.condition:
            self.condition.notify_all()

        if self.config.io.write:
            if time.time() > self.next_dump_time:
                if self.config.run.jobtype == 'profile' and finished_task.type != 'AnalyseProfileTask':
                    if 'AnalyseProfileTask' not in [queued_task.type for queued_task in self.tasks.queued]:
                        if 'AnalyseProfileTask' not in [ongoing_task.type for ongoing_task in self.tasks.ongoing.values()]:
                            self.push_task(self.get_profile_analysis())
                self.dump_snapshot()
                self.status_update()

                self.next_dump_time = time.time() + self.config.io.snapshot_interval
    
    def dump_snapshot(self) -> None:
        print("Dumping snapshot...")
        for name, data in [("config", self.config), ("state", self.tasks)]:
            backup_filename = os.path.join(self.config.io.dir, f"{name}_backup.pkl")
            with open(backup_filename, 'wb') as file:
                # Write to backup to minimize risk of crash while saving
                pickle.dump(data, file) 
                os.system(f'mv {backup_filename} {os.path.join(self.config.io.dir, f"{name}.pkl")}')

    def status_update(self) -> None:
        """Writes status of all tasks in status.txt"""
        print("Writing status update...")
        pad_id = 4
        pad_tasktype = 20
        pad_time = 30
        pad_emit = pad_id + 2
        
        def get_write_line(task: Type[BaseTask]) -> str:
            emit_id = task.emitted_by if task.emitted_by is not None else '-'
            out = f"\n{task.id:<{pad_id}} {emit_id:<{pad_emit}} {task.type:<{pad_tasktype}} \t {task.time_emit.ctime():<{pad_time}}"
            if task.time_launch is not None:
                out += f" {task.time_launch.ctime():<{pad_time}}"
            if task.time_finish is not None:
                out += f" {task.time_finish.ctime():<{pad_time}}"
            return out

        with open(os.path.join(self.config.io.dir, "status.txt"), 'w') as status_file:
            header = f"\n{'id':<{pad_id}} {'from':<{pad_emit}} {'task type':<{pad_tasktype}} \t {'Time when emitted':<{pad_time}} {'Time when started':<{pad_time}} {'Time when finished':<{pad_time}}"
            status_file.write(f"Status of job '{self.config.io.jobname}'\nJobtype: {self.config.run.jobtype}\nStarted: {self.start_date}\nLast updated: {datetime.now()}")
            status_file.write("\n\n=== FAILED ==============================================")
            status_file.write(header)
            for task_id, task_failed in self.tasks.failed.items():
                status_file.write(get_write_line(task_failed))
            status_file.write("\n\n=== IN PROGRESS ==========================================")
            status_file.write(header)
            for task_id, task_ongoing in self.tasks.ongoing.items():
                status_file.write(get_write_line(task_ongoing))
            status_file.write("\n\n=== QUEUED, READY =========================================")
            status_file.write(header)
            for task in self.tasks.queued:
                status_file.write(get_write_line(task))
            status_file.write("\n\n=== QUEUED, NOT READY ====================================")
            status_file.write(header)
            for task_id, task_unready in self.tasks.unready.items():
                status_file.write(get_write_line(task_unready))
            status_file.write("\n\n=== DONE ================================================")
            status_file.write(header)
            for task_id, task_done in self.tasks.done.items():
                status_file.write(get_write_line(task_done))
            status_file.write("\n")
    
    def get_profile_analysis(self) -> AnalyseProfileTask:
        required_tasks = [id for id, task in self.tasks.done.items() if task.type == 'OptimiseTask' or task.type == 'InitialiseOptimiserTask']
        return AnalyseProfileTask(self.config, required_tasks)
    
class SerialContext:
    class MockFuture:
        def __init__(self, finished_task: Type[BaseTask]) -> None:
            self.finished_task = finished_task

        def add_done_callback(self, finalize_func) -> None:
            finalize_func(self)

        def result(self) -> Type[BaseTask]:
            return self.finished_task

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        return True

    def submit(self, run_func, *args) -> MockFuture:
        finished_task = run_func(*args)
        return self.MockFuture(finished_task)

