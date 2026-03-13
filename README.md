# Running the Application

1. Open a Terminal or Windows Powershell
2. Install uv
  -- For Windows: ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```
  -- For Linix/Mac: ```curl -LsSf https://astral.sh/uv/install.sh | sh```
3. Clone the GitHub repository
  -- Navigate to the GitHub repository [https://github.com/serich17/JMLT-work](https://github.com/serich17/JMLT-work)
  -- Select the green <b>Code</b> button and download as zip
  -- Extract the folder where you want in your directory
4. Run the App, and create virtual environment
  -- Navigate to a folder to save the repo to in your file manager
  -- Right click on the folder, and select <b>Open in Terminal</b>
  -- run the following command to start the program ```uv run streamlit run main.py```
  -- The first time it will install all the neccesary packages
5. To close the app:
  -- Return to your terminal and kill. One method is ```CTR C``` multiple times until it stops
  -- Exiting the terminal by itself may not automatically stop the program
  -- Failing to stop the program will continue to use your computer resources