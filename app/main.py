"""
Spaceflight Dynamics — Orbital Simulator (lightweight edition).
Two self-contained tabs: 2D Orbit Simulator · Ground Track (ECEF / ECI).

Authors: De Toni Bernardo, Da Ros Nicola
Run with:  python main.py
"""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from view_2d import GroundTrackView
from view_orbit_sim import OrbitSimulator2D


APP_STYLE = """
* { font-family: 'Inter','Segoe UI Variable','Segoe UI',sans-serif; }
QMainWindow, QWidget { background-color:#0d1117; color:#e6edf3; }
QScrollArea, QScrollArea > QWidget > QWidget { background-color:#0d1117; border:none; }
QGroupBox {
    border:1px solid #30363d; border-radius:8px;
    margin-top:14px; padding:10px 8px 6px 8px;
    background-color:#161b22;
    color:#58a6ff; font-weight:600; font-size:9pt;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left:10px; padding:0 6px; background-color:#0d1117;
}
QLabel { color:#e6edf3; font-size:9pt; }
QLabel#title    { color:#58a6ff; font-size:15pt; font-weight:700; letter-spacing:0.3px; }
QLabel#subtitle { color:#8b949e; font-size:9pt; }
QLabel#credits  { color:#6e7681; font-size:8pt; font-style:italic; }
QLabel#value    { color:#e3b341; font-family:'JetBrains Mono','Consolas',monospace;
                  font-weight:600; }
QLabel#info     { color:#c9d1d9; font-family:'JetBrains Mono','Consolas',monospace;
                  font-size:8pt; }
QPushButton {
    background-color:#21262d; color:#e6edf3;
    border:1px solid #363b42; border-radius:6px;
    padding:6px 14px; font-weight:600; font-size:9pt;
}
QPushButton:hover   { background-color:#30363d; border-color:#58a6ff; }
QPushButton:pressed { background-color:#1f6feb; color:#ffffff; }
QPushButton#play    { background-color:#238636; border-color:#2ea043; color:#ffffff; }
QPushButton#play:hover  { background-color:#2ea043; }
QPushButton#pause   { background-color:#9e6a03; border-color:#bb8009; color:#ffffff; }
QPushButton#pause:hover { background-color:#bb8009; }
QComboBox, QDoubleSpinBox, QSpinBox {
    background-color:#0d1117; color:#e6edf3;
    border:1px solid #30363d; border-radius:5px;
    padding:4px 6px; font-size:9pt;
    selection-background-color:#1f6feb;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color:#58a6ff; }
QComboBox QAbstractItemView {
    background-color:#161b22; color:#e6edf3;
    selection-background-color:#1f6feb; border:1px solid #30363d;
}
QSlider::groove:horizontal { background:#21262d; height:4px; border-radius:2px; }
QSlider::handle:horizontal { background:#58a6ff; width:14px; margin:-6px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#1f6feb; border-radius:2px; }
QTabWidget::pane {
    border:1px solid #30363d; background:#0d1117;
    border-top-left-radius:0; border-top-right-radius:6px;
}
QTabBar::tab {
    background:#0d1117; color:#8b949e;
    padding:9px 20px; border:1px solid transparent; border-bottom:none;
    border-top-left-radius:6px; border-top-right-radius:6px;
    font-weight:600; font-size:9pt; min-width:140px;
}
QTabBar::tab:hover    { color:#e6edf3; }
QTabBar::tab:selected {
    background:#161b22; color:#58a6ff;
    border:1px solid #30363d; border-bottom:1px solid #161b22;
}
QTextEdit {
    background-color:#010409; color:#c9d1d9;
    border:1px solid #30363d; border-radius:6px;
    font-family:'JetBrains Mono','Consolas',monospace; font-size:8pt;
}
QCheckBox, QRadioButton { spacing:8px; font-size:9pt; }
QCheckBox::indicator, QRadioButton::indicator {
    width:16px; height:16px; border:1px solid #30363d; border-radius:4px;
    background:#0d1117;
}
QRadioButton::indicator { border-radius:8px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color:#58a6ff; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background:#1f6feb; border:1px solid #58a6ff;
}
QScrollBar:vertical { background:#0d1117; width:10px; border:none; }
QScrollBar::handle:vertical { background:#30363d; border-radius:5px; min-height:25px; }
QScrollBar::handle:vertical:hover { background:#484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QFrame#separator { background:#30363d; max-height:1px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spaceflight Dynamics — Orbital Simulator')
        self.resize(1400, 860)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(OrbitSimulator2D(),  '2D Orbit Simulator')
        tabs.addTab(GroundTrackView(),   'Ground Track')
        tabs.setCurrentIndex(0)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    families = set(QFontDatabase().families())
    for fam in ('Inter', 'Segoe UI Variable', 'Segoe UI',
                'SF Pro Text', 'Helvetica Neue'):
        if fam in families:
            app.setFont(QFont(fam, 9))
            break
    else:
        app.setFont(QFont('Sans Serif', 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
