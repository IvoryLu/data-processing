print("Once upon a time...")
######
# TREENODE CLASS
######
class TreeNode:
  def __init__(self, story_piece):
    self.story_piece = story_piece
    self.choices = []
  def add_child(self, node):
    self.choices.append(node)
    
  def traverse(self):
    story_node = self
    print(story_node.story_piece)
    while story_node.choices:
      choice = input("Enter 1 or 2 to continue the story: ")
      if int(choice) not in [1, 2]:
        continue
      chosen_index = int(choice)
      chosen_index = chosen_index - 1
      chosen_child = story_node.choices[chosen_index]
      print(chosen_child.story_piece)
      story_node = chosen_child
      
######
# VARIABLES FOR TREE
######
my_story = """
You are in a forest clearing. There is a path to the left.
A bear emerges from the trees and roars!
Do you: 
1 ) Roar back!
2 ) Run to the left...
"""
piece_2 = """
The bear is startled and runs away.
Do you:
1 ) Shout 'Sorry bear!'
2 ) Yell 'Hooray!'
"""
piece_3 = """
The bear is startled and runs away.
Do you:
1 ) Shout 'Sorry bear!'
2 ) Yell 'Hooray!'
"""
piece_4 = """
The bear returns and tells you it's been a rough week. After making peace with
a talking bear, he shows you the way out of the forest.

YOU HAVE ESCAPED THE WILDERNESS.
"""

piece_5 = """
The bear returns and tells you that bullying is not okay before leaving you alone
in the wilderness.

YOU REMAIN LOST.
"""

piece_6 = """
The bear is unamused. After smelling the flowers, it turns around and leaves you alone.

YOU REMAIN LOST.
"""
piece_7 = """
The bear understands and apologizes for startling you. Your new friend shows you a 
path leading out of the forest.

YOU HAVE ESCAPED THE WILDERNESS.
"""
story_root = TreeNode(my_story)
user_choice = input("What is your name? ")
print(user_choice)

choice_a = TreeNode(piece_2)
choice_b = TreeNode(piece_3)
choice_a_1 = TreeNode(piece_4)
choice_a_2 = TreeNode(piece_5)
choice_b_1 = TreeNode(piece_6)
choice_b_2 = TreeNode(piece_7)

story_root.add_child(choice_a)
story_root.add_child(choice_b)
choice_a.add_child(choice_a_1)
choice_a.add_child(choice_a_2)
choice_b.add_child(choice_b_1)
choice_b.add_child(choice_b_2)
story_root.traverse()

######
# TESTING AREA
######
#print(story_root.story_piece)
