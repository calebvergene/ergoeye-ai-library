https://github.com/user-attachments/assets/2fae9a81-d17e-4567-9ec9-1e071af88040

# ErgoEye

This library automates the Rapid Entire Body Assessment (REBA) to rapidly evaluate the risk of musculoskeletal disorders (MSD), addressing the $180 billion global problem of workplace musculoskeletal disorders through AI-powered ergonomic assessment. 

Traditional ergonomic evaluations are outdated, operator-dependent, and reactive only identifying issues after injuries occur.

### What It Does

ErgoEye uses computer vision to analyze worker posture in real-time, automating the industry-standard REBA assessment scale. The system offers two analysis modes:

- **Video Analysis** - Upload recorded videos for comprehensive posture evaluation
- **Real-time Monitoring** - Live posture analysis with real-time data processing

The AI identifies ergonomic risks, compares movements against industry standards, and generates personalized recommendations to help workers improve their posture and prevent injuries before they occur.

## Local Dependencies

This project requires the following packages:

- **OpenCV** - Computer vision library: used for processing video frames
- **MediaPipe** - Machine learning framework for calculating joint positions

```bash
pip install opencv-python
pip install mediapipe
```

To view an implementation of this ergonomic framework, here is a fullstack implementation of the library: 
https://www.youtube.com/watch?v=8jjND9M_Ocw&t=127s
