# main.py
import sys
from slam_manager import SlamManager
from analysis_manager import AnalysisManager

class RobotOrchestrator:
    """
    Acts as the main controller, delegating tasks based on user input.
    This follows the Single Responsibility Principle, as its only role
    is user interaction and orchestration.
    """
    def __init__(self):
        self.slam_manager = SlamManager()
        self.analysis_manager = AnalysisManager()
        self.choices = {
            "1": {"text": "Run SLAM & Save Map", "action": self.slam_manager.run},
            "2": {"text": "Analyze Saved Map", "action": self.analysis_manager.run},
            "3": {"text": "Exit", "action": self.exit_program},
        }

    def display_menu(self):
        """Prints the main menu to the console."""
        print("\n" + "="*25)
        print("  Main Control Menu")
        print("="*25)
        for key, value in self.choices.items():
            print(f"  {key}. {value['text']}")

    def run(self):
        """Runs the main application loop."""
        while True:
            self.display_menu()
            choice = input("Enter your choice: ")
            action_item = self.choices.get(choice)
            if action_item:
                action_item["action"]()
            else:
                print("Invalid choice, please try again.")

    def exit_program(self):
        """Exits the application."""
        print("Exiting.")
        sys.exit(0)

if __name__ == '__main__':
    orchestrator = RobotOrchestrator()
    orchestrator.run()