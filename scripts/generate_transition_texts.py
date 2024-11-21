transitions = {
    "thinking": [
        "Hmm... let me think about what happens next in our story...",
        "I'm imagining the next part of our adventure...",
        "Let's see what magical things happen next...",
        "(soft chime) Taking a moment to dream up what happens next..."
    ],
    
    "continue": [
        "And our story continues...",
        "Let's see what happens next in our tale...",
        "(gentle swoosh) The story flows onward...",
        "Now, where were we? Ah yes..."
    ],
    
    "question": [
        "What do you think might happen next?",
        "I wonder what you think about that?",
        "Would you like to guess what happens next?",
        "(soft bell) What do you imagine will happen?"
    ],
    
    "waiting": [
        "(soft music) I'm creating a special story just for you...",
        "Let me weave together a magical tale...",
        "Getting the next part of our story ready...",
        "(gentle chimes) Preparing something wonderful..."
    ],
    
    "complete": [
        "(soft wind chimes) We've reached a cozy stopping point in our story...",
        "Time for a little rest in our tale...",
        "Let's take a peaceful pause in our story...",
        "(gentle bell) What a lovely adventure we've had so far..."
    ],
    
    "error": [
        "Oops, I lost my place in the story. Let me find it again...",
        "The pages got a bit mixed up. One moment please...",
        "(soft tone) Let me straighten out these story pages...",
        "The story magic needs a moment to settle..."
    ]
}

# Print formatted output
print("Transition Audio Text Content:")
print("=============================")
for category, messages in transitions.items():
    print(f"\n{category.upper()}:")
    for i, message in enumerate(messages, 1):
        print(f"{i}. {message}") 