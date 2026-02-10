import time
from rich.console import Console

console = Console()


class PipelineRagDummy:
    @staticmethod
    def initialize():
        console.print("Initializing RAGs...", style="yellow")
        time.sleep(3)
        console.print("All RAGs are operational.", style="green")
    
    @staticmethod
    def ask(question: str) -> str:
        return f"Dummy answer for: {question}"