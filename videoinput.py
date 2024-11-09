import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading

class VideoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analysis App")
        self.root.geometry("800x600")

        # Set up notebook (tab control)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        # Set up the two pages
        self.video_page = ttk.Frame(self.notebook)
        self.report_page = ttk.Frame(self.notebook)

        self.notebook.add(self.video_page, text="Video Input")
        self.notebook.add(self.report_page, text="Analysis Reports")

        # Initialize components on both pages
        self.setup_video_page()
        self.setup_report_page()

    def setup_video_page(self):
        # Add video display frame with a max size
        self.video_frame = ttk.LabelFrame(self.video_page, text="Video Input")
        self.video_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        # Define max dimensions for video display
        self.max_video_width = 640
        self.max_video_height = 360

        # Add control buttons
        control_frame = ttk.Frame(self.video_page)
        control_frame.pack(pady=10)

        self.upload_button = ttk.Button(control_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(side="left", padx=5)

        self.start_button = ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis, state="disabled")
        self.start_button.pack(side="left", padx=5)

        self.cap = None
        self.video_path = None
        self.video_running = False

    def upload_video(self):
        # File dialog to select a video file
        self.video_path = filedialog.askopenfilename(title="Select Video File",
                                                     filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if self.video_path:
            self.start_button.config(state="normal")
            self.show_video(self.video_path)

    def show_video(self, video_path):
        # Display video frames in the UI
        self.cap = cv2.VideoCapture(video_path)
        self.video_running = True
        self.update_video_frame()

    def update_video_frame(self):
        # Update the video frame in the UI, resizing if necessary
        if self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Get the original frame dimensions
                frame_height, frame_width = frame.shape[:2]

                # Calculate the scale factor to fit within max dimensions
                scale_w = self.max_video_width / frame_width
                scale_h = self.max_video_height / frame_height
                scale = min(scale_w, scale_h, 1.0)  # Limit scaling to avoid enlarging small videos

                # Resize frame according to calculated scale
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to ImageTk format
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Schedule the next frame update
                self.root.after(30, self.update_video_frame)
            else:
                self.video_running = False
                self.cap.release()

    def start_analysis(self):
        if self.video_path:
            # Simulate analysis in a separate thread to avoid freezing the UI
            analysis_thread = threading.Thread(target=self.perform_analysis)
            analysis_thread.start()
        else:
            messagebox.showwarning("No Video", "Please upload a video to analyze.")

    def perform_analysis(self):
        # Simulated video analysis process
        messagebox.showinfo("Analysis", "Video analysis in progress...")
        
        # Example: Updating report table with dummy data
        for i in range(5):
            self.report_table.insert("", "end", values=(f"Report {i+1}", f"Result {i+1}", "Details"))

    def setup_report_page(self):
        # Table to display analysis reports
        self.report_table = ttk.Treeview(self.report_page, columns=("report_id", "result", "details"), show="headings")
        self.report_table.heading("report_id", text="Report ID")
        self.report_table.heading("result", text="Result")
        self.report_table.heading("details", text="Details")

        self.report_table.pack(fill="both", expand=True, padx=10, pady=10)

# Create the main application window
root = tk.Tk()
app = VideoAnalysisApp(root)
root.mainloop()
