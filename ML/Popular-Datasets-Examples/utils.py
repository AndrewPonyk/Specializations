import sys
import time

def animate_print(text, delay=0.1):
    """Helper function to animate each line with proper spacing"""
    try:
        for line in text.split('\n'):
            print(line)
            time.sleep(delay)
        print()  # Ensure a newline after printing
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
        sys.exit(1)

def print_header():
    """Prints the main title of your project"""
    header = """
██████╗  █████╗ ███╗   ███╗
██╔══██╗██╔══██╗████╗ ████║
██████╔╝██████║██╔████╔██║
██╔═══╝ ██╔══██║██║╚██╔╝██║
███████╗╚════██║██║ ╚═╝ ██║
╚══════╝     ╚═╝╚═╝     ╚═╝

welcome to ML Project Hub, where datasets come to life and models are made.
"""

def print_datetime():
    """Prints the current date and time in a clean format."""
    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Date and Time: {current_datetime}")

if __name__ == "__main__":
    print_header()
    print("\n")  # Add a blank line before welcoming
    print("Welcome to ML Project Hub, where datasets come to life and models are made.")
    print_datetime()  # Added this line to call the new function


def print_welcome():
    """Prints a welcome message to your project"""
    welcome = """
Welcome to ML Projects Hub!
where datasets come to life and models are made
"""
    animate_print(welcome, delay=0.1)

if __name__ == "__main__":
    print_header()
    print_welcome()
