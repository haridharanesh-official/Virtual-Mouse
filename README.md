# Virtual Mouse using Hand Gestures

This project uses a webcam, OpenCV, Mediapipe, and PyAutoGUI to control the mouse pointer through hand gestures. It is compatible with Windows, macOS, Linux, and Raspberry Pi.

---

## Features

1. **Virtual Mouse Control**:

   - Move the mouse pointer using the tip of the index finger.
   - Perform left-click by bringing the thumb and index finger close together.
   - Scroll by adjusting the distance between the thumb and index finger.

2. **Cross-Platform Compatibility**:

   - Works on Windows, macOS, Linux, and Raspberry Pi.

3. **Smooth Cursor Movement**:

   - Smoothing factor applied to minimize jittery movements.

4. **Background Task**:

   - The virtual mouse runs in a background thread, allowing other tasks to execute concurrently.

---

## Installation

### Prerequisites

- Python 3.7+
- Webcam (built-in or external)

### Required Libraries

Install the required libraries using pip:

```bash
pip install mediapipe pyautogui opencv-python
```

For Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-opencv
pip3 install mediapipe pyautogui
```

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/virtual-mouse.git
   cd virtual-mouse
   ```

2. Run the script:

   ```bash
   python virtual_mouse.py
   ```

3. Press `q` to quit the application.

---

## Code Structure

- **virtual_mouse.py**: Main script for running the virtual mouse application.
- **README.md**: Documentation for the project.

---

## Features in Detail

- **Cursor Movement**:

  - The index finger's tip controls the mouse cursor position.
  - Movements are smoothed using a configurable `smoothing_factor`.

- **Click Gesture**:

  - Thumb and index finger distance < 20 pixels triggers a left-click.

- **Scroll Gesture**:

  - Thumb and index finger distance > 20 pixels triggers a scroll action.

---

## Compatibility

| Platform     | Supported | Notes                         |
| ------------ | --------- | ----------------------------- |
| Windows      | Yes       | Full support                  |
| macOS        | Yes       | Full support                  |
| Linux        | Yes       | Full support                  |
| Raspberry Pi | Yes       | Optimized for headless setups |

---

## Contributing

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License.

---

## Troubleshooting

- **Webcam not detected**:

  - Ensure your webcam is connected and accessible.
  - Check if another application is using the webcam.

- **High latency on Raspberry Pi**:

  - Reduce the resolution of the camera feed for better performance.
  - Use a lightweight desktop environment or run in headless mode.

---

## Future Enhancements

1. Add support for custom gestures (e.g., right-click, drag-and-drop).
2. Implement a calibration mode for gesture thresholds.
3. Optimize performance for low-power devices like Raspberry Pi.

---

## Acknowledgments

- [Mediapipe](https://mediapipe.dev/): Used for hand landmark detection.
- [PyAutoGUI](https://pyautogui.readthedocs.io/): Used for simulating mouse actions.

---

## Author

[Hari Dharanesh SP](https://github.com/<your-username>)

