import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QMessageBox,
                             QFrame, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import qdarkstyle

import stitch

class VideoProcessor(QThread):
    """视频处理工作线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(bool, str)
    
    def __init__(self, video_path, param1, param2):
        super().__init__()
        self.video_path = video_path
        self.param1 = param1
        self.param2 = param2
        
    def run(self):
        try:
            # 模拟视频处理过程
            for i in range(101):
                self.progress_updated.emit(i)
                self.msleep(50)  # 模拟处理延迟
                
            # 处理完成
            self.processing_finished.emit(True, "视频处理完成！")
        except Exception as e:
            self.processing_finished.emit(False, f"处理失败: {str(e)}")

class VideoProcessingApp(QMainWindow):
    """视频处理主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高级视频处理工具")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        
        self.init_ui()
        
        self.processor = None

    def browse_video0(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择大Fov视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.bigFov_path = file_path
            self.video0_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择大Fov视频: {os.path.basename(file_path)}")

    def browse_video1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择左上视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.left_up_path = file_path
            self.video1_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择左上视频: {os.path.basename(file_path)}")

    def browse_video2(self):
        """浏览并选择右上视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择右上视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        print(file_path)
        
        if file_path:
            self.right_up_path = file_path
            self.video2_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择右上视频: {os.path.basename(file_path)}")

    def browse_video3(self):
        """浏览并选择左下视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择左下视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.left_bottom_path = file_path
            self.video3_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择左上视频: {os.path.basename(file_path)}")

    def browse_video4(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择右下视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.right_bottom_path = file_path
            self.video4_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择右下视频: {os.path.basename(file_path)}")
        
    def init_ui(self):
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # 标题区域
        title_frame = QFrame()
        title_frame.setFrameShape(QFrame.NoFrame)
        title_layout = QHBoxLayout(title_frame)
        
        title_label = QLabel("视频融合工具")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #3498db;")
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        main_layout.addWidget(title_frame)
        path_edit_style = """
            QLineEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 5px;
                padding: 8px;
            }
        """

        button_style="""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
        """
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)


        videos_frame0 = QFrame()
        videos_frame0.setFrameShape(QFrame.StyledPanel)
        videos_frame0.setStyleSheet("background-color: #2c3e50; border-radius: 10px;")
        videos_layout0 = QHBoxLayout(videos_frame0)  # 使用水平布局
        videos_layout0.setContentsMargins(20, 20, 20, 20)
        videos_layout0.setSpacing(20)  # 设置两个区域之间的间距
        # 大Fov视频选择区域
        video0_layout = QVBoxLayout()
        video0_label = QLabel("选择大Fov视频:")
        video0_label.setFont(QFont("Arial", 14))
        video0_label.setStyleSheet("color: #ecf0f1;")
        video0_layout.addWidget(video0_label)

        self.video0_path_edit = QLineEdit()
        self.video0_path_edit.setPlaceholderText("请选择大Fov视频...")
        self.video0_path_edit.setReadOnly(True)
        self.video0_path_edit.setStyleSheet(path_edit_style)
        video0_layout.addWidget(self.video0_path_edit)

        browse0_button = QPushButton("浏览...")
        browse0_button.setStyleSheet(button_style)
        browse0_button.setFixedHeight(35)
        browse0_button.clicked.connect(self.browse_video0)
        video0_layout.addWidget(browse0_button)

        # 将两个区域添加到水平布局中
        videos_layout0.addLayout(video0_layout)

        main_layout.addWidget(videos_frame0)


        # 视频选择区域  2 * 2
        videos_frame = QFrame()
        videos_frame.setFrameShape(QFrame.StyledPanel)
        videos_frame.setStyleSheet("background-color: #2c3e50; border-radius: 10px;")
        videos_layout = QHBoxLayout(videos_frame)  # 使用水平布局
        videos_layout.setContentsMargins(20, 20, 20, 20)
        videos_layout.setSpacing(20)  # 设置两个区域之间的间距

 

        # 左上视频选择区域
        video1_layout = QVBoxLayout()
        video1_label = QLabel("选择左上视频:")
        video1_label.setFont(QFont("Arial", 14))
        video1_label.setStyleSheet("color: #ecf0f1;")
        video1_layout.addWidget(video1_label)

        self.video1_path_edit = QLineEdit()
        self.video1_path_edit.setPlaceholderText("请选择左上视频...")
        self.video1_path_edit.setReadOnly(True)
        self.video1_path_edit.setStyleSheet(path_edit_style)
        video1_layout.addWidget(self.video1_path_edit)

        browse1_button = QPushButton("浏览...")
        browse1_button.setStyleSheet(button_style)
        browse1_button.setFixedHeight(35)
        browse1_button.clicked.connect(self.browse_video1)
        video1_layout.addWidget(browse1_button)

        # 右上视频选择区域
        video2_layout = QVBoxLayout()
        video2_label = QLabel("选择右上视频:")
        video2_label.setFont(QFont("Arial", 14))
        video2_label.setStyleSheet("color: #ecf0f1;")
        video2_layout.addWidget(video2_label)

        self.video2_path_edit = QLineEdit()
        self.video2_path_edit.setPlaceholderText("请选择右上视频...")
        self.video2_path_edit.setReadOnly(True)
        self.video2_path_edit.setStyleSheet(path_edit_style)
        video2_layout.addWidget(self.video2_path_edit)

        browse2_button = QPushButton("浏览...")
        browse2_button.setStyleSheet(button_style)
        browse2_button.setFixedHeight(35)
        browse2_button.clicked.connect(self.browse_video2)
        video2_layout.addWidget(browse2_button)

        # 将两个区域添加到水平布局中
        videos_layout.addLayout(video1_layout)
        videos_layout.addLayout(video2_layout)

        main_layout.addWidget(videos_frame)


        videos_frame2 = QFrame()
        videos_frame2.setFrameShape(QFrame.StyledPanel)
        videos_frame2.setStyleSheet("background-color: #2c3e50; border-radius: 10px;")
        videos_layout2 = QHBoxLayout(videos_frame2)  # 使用水平布局
        videos_layout2.setContentsMargins(20, 20, 20, 20)
        videos_layout2.setSpacing(20)  # 设置两个区域之间的间距


        # 左下视频选择区域
        video3_layout = QVBoxLayout()
        video3_label = QLabel("选择左下视频:")
        video3_label.setFont(QFont("Arial", 14))
        video3_label.setStyleSheet("color: #ecf0f1;")
        video3_layout.addWidget(video3_label)

        self.video3_path_edit = QLineEdit()
        self.video3_path_edit.setPlaceholderText("请选择左下视频...")
        self.video3_path_edit.setReadOnly(True)
        self.video3_path_edit.setStyleSheet(path_edit_style)
        video3_layout.addWidget(self.video3_path_edit)

        browse3_button = QPushButton("浏览...")
        browse3_button.setStyleSheet(button_style)
        browse3_button.setFixedHeight(35)
        browse3_button.clicked.connect(self.browse_video3)
        video3_layout.addWidget(browse3_button)

        # 右下视频选择区域
        video4_layout = QVBoxLayout()
        video4_label = QLabel("选择右下视频:")
        video4_label.setFont(QFont("Arial", 14))
        video4_label.setStyleSheet("color: #ecf0f1;")
        video4_layout.addWidget(video4_label)

        self.video4_path_edit = QLineEdit()
        self.video4_path_edit.setPlaceholderText("请选择右下视频...")
        self.video4_path_edit.setReadOnly(True)
        self.video4_path_edit.setStyleSheet(path_edit_style)
        video4_layout.addWidget(self.video4_path_edit)

        browse4_button = QPushButton("浏览...")
        browse4_button.setStyleSheet(button_style)
        browse4_button.setFixedHeight(35)
        browse4_button.clicked.connect(self.browse_video4)
        video4_layout.addWidget(browse4_button)

        # 将两个区域添加到水平布局中
        videos_layout2.addLayout(video3_layout)
        videos_layout2.addLayout(video4_layout)

        main_layout.addWidget(videos_frame2)
        
        # 按钮区域
        buttons_frame = QFrame()
        buttons_frame.setFrameShape(QFrame.NoFrame)
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setSpacing(20)
        
        self.start_button = QPushButton("开始融合")
        self.start_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #16a085;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.start_button.setFixedHeight(50)
        self.start_button.clicked.connect(self.start_processing)
        buttons_layout.addWidget(self.start_button)
        
        cancel_button = QPushButton("取消")
        cancel_button.setFont(QFont("Arial", 14))
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 8px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #9b59b6;
            }
        """)
        cancel_button.setFixedHeight(50)
        cancel_button.clicked.connect(self.close)
        buttons_layout.addWidget(cancel_button)
        
        main_layout.addWidget(buttons_frame)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def browse_video(self):
        """浏览并选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path_edit.setText(file_path)
            self.statusBar().showMessage(f"已选择视频: {os.path.basename(file_path)}")
    
    def start_processing(self):
        stitch.output_button(self.left_up_path, self.bigFov_path, self.right_up_path, self.left_bottom_path, self.right_bottom_path)
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def processing_done(self, success, message):
        """处理完成回调"""
        self.start_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "失败", message)
        
        self.statusBar().showMessage("处理完成")
        self.processor = None

if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # 创建并显示窗口
    window = VideoProcessingApp()
    window.show()
    
    # 进入应用主循环
    sys.exit(app.exec_())