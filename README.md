# 🧠 Footfall Counter using Computer Vision

### 👩‍💻 Developer: Anjali Gupta  
### 🏫 CMR Institute of Technology, Bengaluru  
### 📅 Submission Date: 3rd November 2025  

---

## 🎯 **Project Objective**
To design an AI-based **Footfall Counter System** that automatically detects and counts the number of people entering and exiting a specific area using **Computer Vision** techniques.

This project demonstrates real-world applications of AI in **retail analytics, crowd monitoring, and smart surveillance** systems.

---

## ⚙️ **Technologies & Libraries Used**

| Component | Tool / Library |
|------------|----------------|
| Programming Language | Python 3.8+ |
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | DeepSORT |
| Image Processing | OpenCV |
| Numerical Computation | NumPy |

---

## 🧩 **Approach**

1. **Video Input:**  
   The system takes a video (`people_footage.mp4`) showing people walking through a particular area.

2. **Detection:**  
   YOLOv8 detects human figures frame by frame.

3. **Tracking:**  
   DeepSORT assigns a unique ID to each person and tracks their movement across frames.

4. **Counting Logic:**  
   A **virtual line** is defined in the frame.  
   - If a person crosses upward → counted as **Entry**  
   - If crossed downward → counted as **Exit**

5. **Output:**  
   - Bounding boxes drawn around each person  
   - Real-time display of total **Entries** and **Exits**  
   - Processed video saved as `output.mp4`

---

## 🧮 **Counting Logic (Visualization)**

