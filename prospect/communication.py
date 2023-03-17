import heapq
import pickle
import time
import traceback
from collections import defaultdict
from typing import Type
from dataclasses import dataclass
from prospect.tasks import BaseTask, initialize_tasks
from multiprocessing import Condition

@dataclass
class TasksState:
    queued: list[Type[BaseTask]]
    # All task dicts are indexed by the task ids 
    unready: dict[int, Type[BaseTask]]
    ongoing: dict[int, Type[BaseTask]]
    done:    dict[int, Type[BaseTask]]
    dependencies: defaultdict[int, list[int]]

class Scheduler:
    def __init__(self, config: dict, state: TasksState = None) -> None:
        self.config = config
        self.next_dump_time = time.time() + config['snapshot_interval']
        self.sleep_iteration = 0
        if state:
            # Resume from snapshot
            self.tasks = state
        else:
            # Start from config file
            self.tasks = TasksState([], {}, {}, {}, defaultdict(list))
            new_tasks = initialize_tasks(self.config)
            for task in new_tasks:
                self.push_task(task)
        self.condition = Condition()

    def delegate(self, executor) -> None:
        # Executor is a context managed instance of either mpi4py.futures.MPICommExecutor 
        # or concurrent.futures.ThreadPoolExecutor
        while True:
            if not self.tasks.queued:
                if not self.tasks.unready:
                    if not self.tasks.ongoing:
                        break

                with self.condition:
                    self.condition.wait()
                continue

            ready_task = self.pop_task()
            args = [self.tasks.done[idx].data for idx in ready_task.required_task_ids]
            executor.submit(ready_task.run_return_self, *args).add_done_callback(self.finalize_task)

            self.sleep_iteration = 0

    def push_task(self, task: Type[BaseTask]) -> None:
        # Pushes task and sets dependencies
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
        self.tasks.ongoing[ready_task.id] = ready_task
        return ready_task

    def is_task_ready(self, task: Type[BaseTask]) -> bool:
        for req in task.required_task_ids:
            if req not in self.tasks.done:
                return False 
        return True

    def finalize_task(self, future) -> None:
        finished_task = future.result() # BaseTask.run_return_self returns the task that was run 
        del self.tasks.ongoing[finished_task.id]
        self.tasks.done[finished_task.id] = finished_task

        # Add tasks emitted by finished_task
        for new_task in finished_task.emit_tasks():
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

        if self.config['write_output']:
            if time.time() > self.next_dump_time:
                dump_snapshot(f"{self.config['output_dir']}/state.pkl", self.tasks)
                status_update(f"{self.config['output_dir']}/status.txt", self.tasks)
                self.next_dump_time = time.time() + self.config['snapshot_interval']

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

def dump_snapshot(filename: str, state: TasksState) -> None:
    print("Dumping snapshot...")
    with open(filename, 'wb') as state_file:
        pickle.dump(state, state_file)

def status_update(filename: str, state: TasksState) -> None:
    print("Writing status update... (to be implemented)")
