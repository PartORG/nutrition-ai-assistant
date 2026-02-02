# Nutrition AI Assistant

## Setting Up the Environment

Follow these steps to set up the development environment for the Nutrition AI Assistant:

1. **Install Python**
   - Ensure you have Python 3.11.3 installed. You can use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:
     ```bash
     pyenv install 3.11.3
     pyenv local 3.11.3
     ```

2. **Create a Virtual Environment**
   - Create and activate a virtual environment:
     ```bash
     python -m venv .venv
     # On Windows
     .venv\Scripts\Activate.ps1
     # On macOS/Linux
     source .venv/bin/activate
     ```

3. **Upgrade pip**
   - Upgrade pip to the latest version:
     ```bash
     python -m pip install --upgrade pip
     ```

4. **Install Dependencies**
   - Install the required dependencies:
     ```bash
     pip install -r requirements-dev.txt
     ```

5. **Verify Installation**
   - Ensure all dependencies are installed correctly by running:
     ```bash
     pip list
     ```

You're now ready to start working on the Nutrition AI Assistant!