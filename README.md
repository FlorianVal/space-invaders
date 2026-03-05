

# Space Invaders with Custom Controls

This project allows you to play Space Invaders with custom input methods using a WebSocket bridge. The architecture separates the game from the input method, allowing you to control the game using various interfaces including keyboard, scripts, or even computer vision!

## Setup and Installation

### Installing Node.js

If you don't have Node.js installed, follow these instructions:

#### Windows
1. Download the installer from [nodejs.org](https://nodejs.org/)
2. Run the installer and follow the installation wizard
3. Verify installation by running `node -v` in Command Prompt or PowerShell

#### macOS
Option 1: Using the installer
1. Download the macOS installer from [nodejs.org](https://nodejs.org/)
2. Run the installer and follow the installation wizard

Option 2: Using Homebrew
```bash
brew install node
```

#### Linux
Ubuntu/Debian:
```bash
sudo apt update
sudo apt install nodejs npm
```

Fedora:
```bash
sudo dnf install nodejs
```

Verify your installation:
```bash
node -v
npm -v
```

### Prerequisites

- Node.js (for WebSocket server)
- Python 3.6+
- Required Python packages: `websockets` (for control module)

```bash
# Install required Python packages
pip install websockets
```

## How to Run the Game

You need to run three separate components to play the game:

### 1. Start the HTTP Server for the Game

```bash
# In the project root directory
python -m http.server 8000
```

This will serve the game at `http://localhost:8000` - open this URL in your browser.

### 2. Start the WebSocket Server

```bash
# In the project root directory
node server.js
```

This starts the WebSocket server on port 8765, which acts as a bridge between your control inputs and the game.

### 3. Start the Control Module

```bash
# In the project root directory
python control_module.py
```

This runs the basic input module that takes keyboard commands and sends them to the game.

## How It Works

1. The Space Invaders game runs in the browser and listens for WebSocket events
2. The Node.js server (`server.js`) acts as a bridge, receiving commands and converting them into keypress events
3. The control module (`control_module.py`) captures your input and sends commands to the WebSocket server

The following commands are available in the basic control module:
- 'q' or 'left' = Move Left
- 'd' or 'right' = Move Right
- 'space' or 'f' = Fire
- 'enter' or 's' = Enter/Select
- 'a' = Quit the control module

## Your Assignment: Computer Vision Controls

Your task is to create a computer vision-based controller for the game. You should:

1. Train a computer vision model that can recognize gestures, objects, or movements
2. Create a new control module based on `control_module.py` that:
   - Captures video from a camera
   - Processes frames with your computer vision model
   - Translates detected actions into game commands
   - Sends those commands to the WebSocket server

For example, you might detect hand positions and map them to the following controls:
- Hand on left side → LEFT command
- Hand on right side → RIGHT command
- Hand raised → FIRE command
- Two hands visible → ENTER command

## Implementation Tips

1. Study the existing `control_module.py` to understand how to send commands to the WebSocket
2. You can use any computer vision library (OpenCV, MediaPipe, etc.)
3. Test your system with simple detection before implementing advanced features
4. Consider what gestures or objects would make intuitive game controls
5. Remember that the game only needs four commands: LEFT, RIGHT, FIRE, and ENTER

Good luck and have fun building your computer vision-controlled Space Invaders game!


## License

Free to use and abuse under the MIT license.
http://www.opensource.org/licenses/mit-license.php
