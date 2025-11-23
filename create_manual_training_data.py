"""
Manual Training Data Creator for Rexx RAG System
This script helps you create high-quality, domain-specific training data
and merge it with automated data for better fine-tuning results.
"""

import json
import os
from datetime import datetime
from typing import List, Dict


class ManualTrainingDataCreator:
    """Helper class for creating and managing manual training data."""
    
    def __init__(self, output_file: str = "manual_training_data.json"):
        self.output_file = output_file
        self.training_data = []
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing manual training data if available."""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"Loaded {len(self.training_data)} existing training examples")
        else:
            print("No existing data found. Starting fresh.")
    
    def add_training_example(self, 
                           question: str, 
                           context: str, 
                           answer: str,
                           source: str = "manual",
                           category: str = "general",
                           difficulty: str = "medium") -> Dict:
        """
        Add a single training example with metadata.
        
        Args:
            question: The question to ask
            context: The relevant context from documentation
            answer: The expected answer (substring from context)
            source: Source document identifier
            category: Question category (e.g., 'configuration', 'features', 'troubleshooting')
            difficulty: Question difficulty (easy, medium, hard)
        
        Returns:
            The created training example
        """
        # Find answer position in context
        answer_start = context.find(answer)
        
        if answer_start == -1:
            print(f"WARNING: Answer not found in context for question: {question}")
            print(f"Answer: {answer[:50]}...")
            print(f"Context: {context[:100]}...")
            answer_start = 0
        
        example = {
            "id": f"manual_{len(self.training_data)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "context": context,
            "question": question,
            "answer": {
                "text": answer,
                "answer_start": answer_start
            },
            "source": source,
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "category": category,
                "difficulty": difficulty,
                "manual": True
            }
        }
        
        self.training_data.append(example)
        print(f"✓ Added example {len(self.training_data)}: {question[:60]}...")
        
        return example
    
    def add_batch_from_template(self, examples: List[Dict]):
        """
        Add multiple examples at once using a template format.
        
        Args:
            examples: List of dicts with 'question', 'context', 'answer', etc.
        """
        for ex in examples:
            self.add_training_example(
                question=ex['question'],
                context=ex['context'],
                answer=ex['answer'],
                source=ex.get('source', 'manual'),
                category=ex.get('category', 'general'),
                difficulty=ex.get('difficulty', 'medium')
            )
    
    def save(self):
        """Save training data to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"\n✓ Saved {len(self.training_data)} examples to {self.output_file}")
    
    def get_statistics(self):
        """Get statistics about the training data."""
        if not self.training_data:
            return "No training data available"
        
        categories = {}
        difficulties = {}
        
        for example in self.training_data:
            metadata = example.get('metadata', {})
            cat = metadata.get('category', 'unknown')
            diff = metadata.get('difficulty', 'unknown')
            
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        stats = {
            "total_examples": len(self.training_data),
            "categories": categories,
            "difficulties": difficulties,
            "avg_question_length": sum(len(ex['question']) for ex in self.training_data) / len(self.training_data),
            "avg_context_length": sum(len(ex['context']) for ex in self.training_data) / len(self.training_data),
            "avg_answer_length": sum(len(ex['answer']['text']) for ex in self.training_data) / len(self.training_data)
        }
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        if isinstance(stats, str):
            print(stats)
            return
        
        print("\n" + "="*60)
        print("TRAINING DATA STATISTICS")
        print("="*60)
        print(f"Total Examples: {stats['total_examples']}")
        print(f"\nAverage Lengths:")
        print(f"  - Question: {stats['avg_question_length']:.1f} chars")
        print(f"  - Context: {stats['avg_context_length']:.1f} chars")
        print(f"  - Answer: {stats['avg_answer_length']:.1f} chars")
        
        print(f"\nCategories:")
        for cat, count in stats['categories'].items():
            print(f"  - {cat}: {count}")
        
        print(f"\nDifficulties:")
        for diff, count in stats['difficulties'].items():
            print(f"  - {diff}: {count}")
        print("="*60)
    
    def merge_with_automated_data(self, automated_file: str = "rexx_training_data.json",
                                  output_file: str = "combined_training_data.json"):
        """
        Merge manual training data with automated data.
        
        Args:
            automated_file: Path to automated training data
            output_file: Output file for combined data
        """
        if not os.path.exists(automated_file):
            print(f"Automated file not found: {automated_file}")
            return
        
        with open(automated_file, 'r') as f:
            automated_data = json.load(f)
        
        # Mark automated data
        for ex in automated_data:
            if 'metadata' not in ex:
                ex['metadata'] = {}
            ex['metadata']['manual'] = False
        
        combined = self.training_data + automated_data
        
        with open(output_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\n✓ Merged {len(self.training_data)} manual + {len(automated_data)} automated")
        print(f"✓ Total: {len(combined)} examples saved to {output_file}")
        
        return combined


# ============================================================================
# EXAMPLE USAGE & TEMPLATES
# ============================================================================

def create_example_training_data():
    """Example function showing how to create high-quality manual training data."""
    
    creator = ManualTrainingDataCreator("manual_training_data.json")
    
    # Example 1: User Management Questions
    rexx_user_examples = [
        {
            "question": "Can two users have the same User Name in Rexx?",
            "context": "In Rexx Systems, each user must have a unique User Name. The system enforces this constraint to ensure proper user identification and access control. Duplicate User Names are not permitted within the same Rexx installation.",
            "answer": "each user must have a unique User Name",
            "source": "User Management Documentation",
            "category": "user_management",
            "difficulty": "easy"
        },
        {
            "question": "What happens if I try to create a user with an existing User Name?",
            "context": "When attempting to create a new user with a User Name that already exists in the system, Rexx will display an error message and prevent the user creation. The administrator must choose a different User Name to proceed with user creation.",
            "answer": "Rexx will display an error message and prevent the user creation",
            "source": "User Management Documentation",
            "category": "user_management",
            "difficulty": "medium"
        },
        {
            "question": "How do I configure user permissions in Rexx?",
            "context": "User permissions in Rexx are configured through Permission Profiles (Berechtigungsprofile). Navigate to Admin > Ticket > Permission Profiles to create or edit profiles. Each user can be assigned one or multiple permission profiles that determine their access rights to different modules and functions.",
            "answer": "User permissions in Rexx are configured through Permission Profiles (Berechtigungsprofile)",
            "source": "Administrator Training Manual",
            "category": "configuration",
            "difficulty": "medium"
        }
    ]
    
    # Example 2: HR Module Questions
    hr_module_examples = [
        {
            "question": "What are the main modules in Rexx HR?",
            "context": "Rexx HR consists of several integrated modules including Personnel Management, Time Management, Recruiting, Training & Development, Performance Management, and Reporting. These modules work together to provide comprehensive HR functionality.",
            "answer": "Personnel Management, Time Management, Recruiting, Training & Development, Performance Management, and Reporting",
            "source": "HR Module Overview",
            "category": "features",
            "difficulty": "easy"
        },
        {
            "question": "How does the organizational tree work in Rexx?",
            "context": "The organizational tree (Org.Tree) in Rexx displays the hierarchical structure of your company. It shows organizational units, positions (Stellen), and employees. Administrators can use drag-and-drop to move positions and employees between organizational units. The tree supports multiple levels including divisions, departments, and teams.",
            "answer": "displays the hierarchical structure of your company. It shows organizational units, positions (Stellen), and employees",
            "source": "Organization Management Guide",
            "category": "features",
            "difficulty": "medium"
        }
    ]
    
    # Example 3: Troubleshooting Questions
    troubleshooting_examples = [
        {
            "question": "What should I do if I forgot my Rexx password?",
            "context": "If you forget your Rexx password, click on the 'Passwort vergessen' (Forgot Password) link on the login page. Enter the email address registered in the system and request a new password. The system will send a password reset link to your registered email address.",
            "answer": "click on the 'Passwort vergessen' (Forgot Password) link on the login page",
            "source": "Login Documentation",
            "category": "troubleshooting",
            "difficulty": "easy"
        },
        {
            "question": "Why can't I see certain menu items in Rexx?",
            "context": "Menu visibility in Rexx is controlled by your assigned Permission Profiles (Berechtigungsprofile). If you cannot see expected menu items, your permission profile may not include the necessary rights. Contact your Rexx administrator to verify your permissions under Admin > Ticket > Permission Profiles.",
            "answer": "Menu visibility in Rexx is controlled by your assigned Permission Profiles",
            "source": "Administrator Training Manual",
            "category": "troubleshooting",
            "difficulty": "medium"
        }
    ]
    
    # Add all examples
    creator.add_batch_from_template(rexx_user_examples)
    creator.add_batch_from_template(hr_module_examples)
    creator.add_batch_from_template(troubleshooting_examples)
    
    # Save and show statistics
    creator.save()
    creator.print_statistics()
    
    return creator


def interactive_mode():
    """Interactive mode for adding training data."""
    creator = ManualTrainingDataCreator("manual_training_data.json")
    
    print("\n" + "="*60)
    print("MANUAL TRAINING DATA CREATOR - Interactive Mode")
    print("="*60)
    print("Create high-quality training examples for your RAG system.")
    print("Type 'quit' at any time to exit and save.\n")
    
    while True:
        print("\n--- New Training Example ---")
        
        question = input("Question: ").strip()
        if question.lower() == 'quit':
            break
        
        context = input("Context (relevant documentation text): ").strip()
        if context.lower() == 'quit':
            break
        
        answer = input("Answer (must be found in context): ").strip()
        if answer.lower() == 'quit':
            break
        
        source = input("Source document (e.g., 'User Manual p.15'): ").strip() or "manual"
        category = input("Category (user_management/features/configuration/troubleshooting): ").strip() or "general"
        difficulty = input("Difficulty (easy/medium/hard): ").strip() or "medium"
        
        creator.add_training_example(
            question=question,
            context=context,
            answer=answer,
            source=source,
            category=category,
            difficulty=difficulty
        )
        
        continue_input = input("\nAdd another example? (y/n): ").strip().lower()
        if continue_input != 'y':
            break
    
    creator.save()
    creator.print_statistics()
    
    # Ask about merging
    merge = input("\nMerge with automated training data? (y/n): ").strip().lower()
    if merge == 'y':
        creator.merge_with_automated_data()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Rexx RAG Manual Training Data Creator")
    print("="*60)
    print("Options:")
    print("1. Create example training data (demonstrates best practices)")
    print("2. Interactive mode (add your own examples)")
    print("3. View existing statistics")
    print("4. Merge manual and automated data")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print("\nCreating example training data...")
        create_example_training_data()
        
    elif choice == "2":
        interactive_mode()
        
    elif choice == "3":
        creator = ManualTrainingDataCreator("manual_training_data.json")
        creator.print_statistics()
        
    elif choice == "4":
        creator = ManualTrainingDataCreator("manual_training_data.json")
        creator.merge_with_automated_data()
        
    else:
        print("Invalid choice")
