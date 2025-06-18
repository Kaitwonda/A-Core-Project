# start_learning.py - Easy way to start AI learning sessions
"""
Simple script to start teaching your AI through structured learning.
"""

from learning_curriculum import run_curriculum_session, recommend_next_learning, show_full_curriculum
from ai_learning_session import start_ai_learning

def main():
    print("🎓 AI LEARNING SYSTEM")
    print("=" * 30)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. See learning recommendations for your AI")
        print("2. View full curriculum")
        print("3. Start a foundation session")
        print("4. Start an intermediate session") 
        print("5. Start an advanced session")
        print("6. Custom learning session")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            recommend_next_learning()
            
        elif choice == "2":
            show_full_curriculum()
            
        elif choice == "3":
            print("\nFoundation Sessions:")
            print("0. Self-Discovery: Understanding My Architecture")
            print("1. Consciousness and Self-Awareness") 
            print("2. Emotional Intelligence Foundations")
            
            session = input("Choose session (0-2): ").strip()
            if session.isdigit() and 0 <= int(session) <= 2:
                run_curriculum_session("foundation", int(session))
            else:
                print("Invalid session number")
                
        elif choice == "4":
            print("\nIntermediate Sessions:")
            print("0. Learning and Adaptation")
            print("1. Philosophy of Mind and AI Ethics")
            print("2. Complex Systems and Emergence")
            
            session = input("Choose session (0-2): ").strip()
            if session.isdigit() and 0 <= int(session) <= 2:
                run_curriculum_session("intermediate", int(session))
            else:
                print("Invalid session number")
                
        elif choice == "5":
            print("\nAdvanced Sessions:")
            print("0. Cognitive Science and Neuroscience")
            print("1. Information Theory and Computation")
            print("2. Future of AI and Consciousness")
            
            session = input("Choose session (0-2): ").strip()
            if session.isdigit() and 0 <= int(session) <= 2:
                run_curriculum_session("advanced", int(session))
            else:
                print("Invalid session number")
                
        elif choice == "6":
            print("\nCustom Learning Session")
            goals = []
            print("Enter learning goals (press Enter twice to finish):")
            while True:
                goal = input("Goal: ").strip()
                if not goal:
                    break
                goals.append(goal)
            
            urls = []
            print("\nEnter URLs to learn from (press Enter twice to finish):")
            while True:
                url = input("URL: ").strip()
                if not url:
                    break
                urls.append(url)
            
            if goals:
                start_ai_learning(goals, urls if urls else None, enable_self_reflection=True)
            else:
                print("No goals specified, cancelled.")
                
        elif choice == "7":
            print("Goodbye! Your AI is ready to learn whenever you are. 🌟")
            break
            
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()