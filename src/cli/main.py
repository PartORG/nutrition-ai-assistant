# Typer + Rich app
# update requirements-dev.txt with typer, rich 
# --help , initialize (PipelineRagDummy.initialize()),
# chat (PipelineRagDummy.ask()),
# dummy class PipelineRAGDummy: 
#    - method initialize(return string "initializing RAGs..." timer.sleep(3) "All RAGS are operational.")
#    - method ask(question) --> question: user_question, Answer: 'Dummy answer'
# 
# CLI chat: while loop that checks if user typed 'exit'/ 'quit'
# user input - stores into question variable and calls PipelineRagDummy.ask(question);
# 
# src/cli/main.py :
# 
# if __name__ == "__main__":
#     cli_app()
# 
# (venv) src/cli/ python 
