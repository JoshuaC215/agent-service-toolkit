import streamlit as st

from schema.task_data import TaskData


class TaskDataStatus:
    def __init__(self) -> None:
        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        status = self.status
        status_str = f"Task **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "has :blue[started]. Input:"
            case "running":
                status_str += "wrote:"
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[completed successfully]. Output:"
                else:
                    status_str += ":red[ended with error]. Output:"
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
        if task_data.run_id not in self.current_task_data:
            # Status label always shows the last newly started task
            status.update(label=f"""Task: {task_data.name}""")
        self.current_task_data[task_data.run_id] = task_data
        # Status is "running" until all tasks have completed
        if not any(entry.completed() for entry in self.current_task_data.values()):
            state = "running"
        # Status is "error" if any task has errored
        elif any(entry.completed_with_error() for entry in self.current_task_data.values()):
            state = "error"
        # Status is "complete" if all tasks have completed successfully
        else:
            state = "complete"
        status.update(state=state)
