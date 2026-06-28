# Generate the README demo GIF with VHS (https://github.com/charmbracelet/vhs):
#   vhs demo.tape        ->  assets/demo.gif
#
# VHS needs: vhs + ffmpeg + ttyd installed. On Windows, the easiest alternative
# is to just run `python demo_gif.py` in a terminal and capture the window with
# ScreenToGif (https://www.screentogif.com) — the script is self-paced for that.

Output assets/demo.gif

Set FontSize 16
Set Width 1200
Set Height 780
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 55ms

Type "python demo_gif.py"
Sleep 500ms
Enter
Sleep 20s
