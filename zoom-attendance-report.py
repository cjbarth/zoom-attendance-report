import os
import sys
from datetime import datetime

import pandas as pd

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

from PySide6.QtGui import QPalette, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDialog,
    QLabel,
    QTimeEdit,
    QDialogButtonBox,
    QFormLayout,
    QComboBox,
)
from PySide6.QtCore import QTime


def normalize_name(name):
    if pd.isna(name):
        return None
    return " ".join(name.lower().strip().split())


def group_names_by_embedding_similarity(names, threshold=0.85):
    if not EMBEDDING_AVAILABLE:
        raise ImportError("sentence-transformers is not installed")

    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    embeddings = model.encode(names, convert_to_tensor=True).cpu()
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(embeddings)
    n = len(names)
    clusters = []
    assigned = set()
    name_to_cluster = {}

    for i in range(n):
        if names[i] in assigned:
            continue
        cluster = [names[i]]
        name_to_cluster[names[i]] = len(clusters)
        for j in range(i + 1, n):
            if names[j] in assigned:
                continue
            if similarity_matrix[i][j] >= threshold:
                cluster.append(names[j])
                assigned.add(names[j])
                name_to_cluster[names[j]] = len(clusters)
        clusters.append(sorted(cluster, key=str.lower))
        assigned.add(names[i])

    return name_to_cluster


def guess_meeting_times(df):
    """Guess the weeknight and weekend meeting times based on the largest attendance."""
    df["hour"] = pd.to_datetime(df["Start time"], errors="coerce").dt.hour
    df["day"] = pd.to_datetime(df["Start time"], errors="coerce").dt.day_name()

    # Filter for evening meetings on weekdays
    weeknight_data = df[
        df["day"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        & df["hour"].between(18, 23)
    ]
    weekend_data = df[df["day"].isin(["Saturday", "Sunday"])]

    # Find the day and hour with the largest attendance for each period
    weeknight_meeting = (
        weeknight_data.groupby(["day", "hour"]).size().idxmax()
        if not weeknight_data.empty
        else ("Tuesday", 19)
    )
    weekend_meeting = (
        weekend_data.groupby(["day", "hour"]).size().idxmax()
        if not weekend_data.empty
        else ("Sunday", 10)
    )

    return weekend_meeting, weeknight_meeting


def filter_major_meetings(df, weekend_meeting, weeknight_meeting):
    """Filter the data to include only the largest weekday evening and weekend meetings."""
    df["corrected_start"] = pd.to_datetime(df["Start time"], errors="coerce")
    df["corrected_day"] = df["corrected_start"].dt.day_name()
    df["corrected_hour"] = df["corrected_start"].dt.hour
    df["corrected_date"] = df[
        "corrected_start"
    ].dt.date  # Ensure corrected_date is created

    # Filter for the specific weekend and weeknight meetings
    is_weekend = (df["corrected_day"] == weekend_meeting[0]) & (
        df["corrected_hour"] == weekend_meeting[1]
    )
    is_weeknight = (df["corrected_day"] == weeknight_meeting[0]) & (
        df["corrected_hour"] == weeknight_meeting[1]
    )

    return df[
        is_weekend | is_weeknight
    ].copy()  # Retain corrected_date in the filtered DataFrame


class MeetingTimeSelector(QDialog):
    """Dialog for selecting weeknight and weekend meeting times."""

    def __init__(self, weekend_meeting, weeknight_meeting, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Meeting Times")

        # Day and time selectors for weekend
        self.weekend_day_combo = QComboBox()
        self.weekend_day_combo.addItems(["Saturday", "Sunday"])
        self.weekend_day_combo.setCurrentText(weekend_meeting[0])

        self.weekend_time_edit = QTimeEdit(QTime(weekend_meeting[1], 0))
        self.weekend_time_edit.setDisplayFormat("h AP")  # Remove leading zero
        self.weekend_time_edit.setMinimumTime(QTime(0, 0))  # Allow only valid hours
        self.weekend_time_edit.setMaximumTime(QTime(23, 0))

        # Day and time selectors for weeknight
        self.weeknight_day_combo = QComboBox()
        self.weeknight_day_combo.addItems(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )
        self.weeknight_day_combo.setCurrentText(weeknight_meeting[0])

        self.weeknight_time_edit = QTimeEdit(QTime(weeknight_meeting[1], 0))
        self.weeknight_time_edit.setDisplayFormat("h AP")  # Remove leading zero
        self.weeknight_time_edit.setMinimumTime(QTime(0, 0))  # Allow only valid hours
        self.weeknight_time_edit.setMaximumTime(QTime(23, 0))

        # Layout
        layout = QFormLayout()
        layout.addRow(QLabel("Weekend Meeting Day:"), self.weekend_day_combo)
        layout.addRow(QLabel("Weekend Meeting Starting Hour:"), self.weekend_time_edit)
        layout.addRow(QLabel("Weeknight Meeting Day:"), self.weeknight_day_combo)
        layout.addRow(
            QLabel("Weeknight Meeting Starting Hour:"), self.weeknight_time_edit
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_meetings(self):
        return (
            (
                self.weekend_day_combo.currentText(),
                self.weekend_time_edit.time().hour(),
            ),
            (
                self.weeknight_day_combo.currentText(),
                self.weeknight_time_edit.time().hour(),
            ),
        )


def main(csv_path):
    df = pd.read_csv(csv_path)

    # Guess meeting times
    weekend_meeting, weeknight_meeting = guess_meeting_times(df)

    # Show UI for user to confirm or adjust meeting times
    app = QApplication.instance() or QApplication(sys.argv)
    selector = MeetingTimeSelector(weekend_meeting, weeknight_meeting)
    if selector.exec() == QDialog.Accepted:
        weekend_meeting, weeknight_meeting = selector.get_meetings()
    else:
        sys.exit(0)

    # Filter data based on selected meeting times
    filtered_df = filter_major_meetings(df, weekend_meeting, weeknight_meeting)

    filtered_df["normalized_name"] = filtered_df["Name (original name)"].apply(
        normalize_name
    )

    name_groups = (
        filtered_df[["normalized_name", "Name (original name)"]]
        .dropna()
        .groupby("normalized_name")["Name (original name)"]
        .agg(lambda names: max(names, key=lambda n: len(str(n))))
    )

    filtered_df["Display Name"] = filtered_df["normalized_name"].map(name_groups)
    filtered_df = filtered_df[filtered_df["Display Name"].notna()]

    if EMBEDDING_AVAILABLE:
        name_to_cluster = group_names_by_embedding_similarity(
            filtered_df["Display Name"]
            .sort_values(key=lambda y: y.str.lower())
            .unique()
            .tolist()
        )
        filtered_df["cluster"] = filtered_df["Display Name"].map(
            lambda x: name_to_cluster.get(x, -1)
        )
    else:
        filtered_df["cluster"] = 0

    date_cols = filtered_df["corrected_date"].unique()
    date_type = {
        date: "■" if datetime.strptime(str(date), "%Y-%m-%d").weekday() > 5 else "●"
        for date in date_cols
    }

    attendance = filtered_df.pivot_table(
        index=["Display Name", "cluster"],
        columns="corrected_date",
        values="Join time",
        aggfunc="count",
    ).notna()

    attendance = attendance.apply(
        lambda col: col.map(lambda v: date_type[col.name] if v else ""), axis=0
    )

    attendance.sort_values(by=["cluster", "Display Name"], inplace=True)
    attendance.index = attendance.index.droplevel("cluster")

    output_path = os.path.splitext(csv_path)[0] + "_Report.csv"
    attendance.to_csv(output_path)
    print(f"Report saved to: {output_path}")

    show_gui(attendance)


def show_gui(df):
    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Attendance Report")

    table = QTableWidget()
    df = df.reset_index()
    rows, cols = df.shape
    table.setRowCount(rows)
    table.setColumnCount(cols)
    table.setHorizontalHeaderLabels(df.columns.astype(str).tolist())

    palette = app.palette()
    base_color = palette.color(QPalette.Base)
    alt_color_1 = base_color.lighter(105)
    alt_color_2 = base_color.darker(108)

    for row in range(rows):
        band_index = (row // 3) % 2
        stripe_color = alt_color_1 if band_index == 0 else alt_color_2

        for col in range(cols):
            val = str(df.iat[row, col])
            item = QTableWidgetItem(val)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(
                Qt.AlignCenter if col > 0 else Qt.AlignLeft | Qt.AlignVCenter
            )
            item.setBackground(stripe_color)
            table.setItem(row, col, item)

    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    layout = QVBoxLayout()
    layout.addWidget(table)

    container = QWidget()
    container.setLayout(layout)
    window.setCentralWidget(container)
    window.resize(1000, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        app = QApplication(sys.argv)
        csv_path, _ = QFileDialog.getOpenFileName(
            None, "Select CSV File", "", "CSV files (*.csv);;All files (*)"
        )
        if not csv_path:
            sys.exit(0)

    main(csv_path)
