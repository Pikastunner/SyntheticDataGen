import os
import subprocess
import shutil
import sys
import time
import threading

def run_command(command):
    def show_status():
        """Displays status text that cycles between 'Executing Utility.' -> 'Executing Utility..' -> 'Executing Utility...'"""
        status_text = "Executing Utility."
        while process.poll() is None:  # While the process is running
            for i in range(3):  # Cycle through . -> .. -> ...
                sys.stdout.write(f"\r{status_text + '.' * i}")
                sys.stdout.flush()
                time.sleep(0.5)  # Pause to give visual effect of cycling

    if os.name == 'nt':  # Check if running on Windows
        if shutil.which("wsl") is None:  # Check if WSL is installed
            print("WSL is not installed. Note that SyntheticImageGen makes use of binaries only available on Linux distributions.")
            print("You can install WSL by running the following command in PowerShell as Administrator: wsl --install")
            return
        else:
            # Prefix command with 'wsl' to run in WSL
            command = f"wsl {command}"

    # Start the subprocess and a separate thread for status
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Start a separate thread to show the status message while the process runs
    status_thread = threading.Thread(target=show_status)
    status_thread.start()

    # Print the output in real-time
    for line in process.stdout:
        sys.stdout.write(line)

    # Capture and print any error messages
    stderr_output = process.stderr.read()
    if stderr_output:
        sys.stderr.write(stderr_output)

    # Wait for the process to complete
    process.wait()
    status_thread.join()  # Ensure status thread ends when process completes

    # Check the return code
    if process.returncode == 0:
        print("\nCommand executed successfully.")
    else:
        print(f"\nError executing command. Return code: {process.returncode}")

# Example usage
# run_command("your_command_here")
