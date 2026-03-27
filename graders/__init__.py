"""Graders for scoring agent performance on each task."""

from .base_grader import grade_episode
from .task1_grader import Task1Grader
from .task2_grader import Task2Grader
from .task3_grader import Task3Grader

GRADERS = {
    "task_1": Task1Grader(),
    "task_2": Task2Grader(),
    "task_3": Task3Grader(),
}

__all__ = ["grade_episode", "Task1Grader", "Task2Grader", "Task3Grader", "GRADERS"]
