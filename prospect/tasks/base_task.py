from abc import ABC, abstractmethod
from datetime import date
import time
from typing import Any
import traceback
from prospect.input import Configuration

class BaseTask(ABC):
    idx_count = -1
    priority: float
    required_task_ids: list[int]
    data: Any # Result from self.run() method is stored here
    success: bool

    time_launch: date = None
    time_emit: date = None
    time_finish: date = None
    emitted_by: int = None

    def __init__(self, config: Configuration, required_task_ids=[]):
        self.config = config
        self.id = BaseTask.idx_count
        if self.id < 0:
            raise ValueError('Tried initializing a task with negative id, probably because emit_tasks was called by a worker process.')
        for req in required_task_ids:
            if req >= self.id:
                raise ValueError(f'Cannot require non-existent task of ID {req}.')
            elif req < 0:
                raise ValueError('Required task ID is negative.')
        self.required_task_ids = required_task_ids
        BaseTask.idx_count += 1

    @abstractmethod
    def run(self) -> None:
        pass

    def emit_tasks(self):
        # Any required information must be stored in self.data during call to run()
        return []

    def run_return_self(self, *args):
        self.success = True
        self.error = None
        try:
            tic = time.perf_counter()
            self.run(*args)
            toc = time.perf_counter()
            print(f"Finished task of type {self.type} and id {self.id} in {toc - tic:.3} seconds")
            self.finalize()
            return self
        except Exception as e:
            self.success = False
            self.error = traceback.format_exc()
            self.finalize()
            return self

    def __lt__(self, other):
        # Largest numerical value of priority is greatest
        return self.priority > other.priority

    def finalize(self):
        pass

    @property
    def type(self) -> str:
        return self.__class__.__name__
