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
)


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


def main(csv_path):
    df = pd.read_csv(csv_path)

    df["corrected_start"] = pd.to_datetime(df["Start time"], errors="coerce")
    df["corrected_day"] = df["corrected_start"].dt.day_name()
    df["corrected_date"] = df["corrected_start"].dt.date
    df["corrected_hour"] = df["corrected_start"].dt.hour

    is_sunday = df["corrected_day"] == "Sunday"
    is_tue_thu_evening = df["corrected_day"].isin(["Tuesday", "Thursday"]) & df[
        "corrected_hour"
    ].between(18, 23)
    filtered_df = df[is_sunday | is_tue_thu_evening].copy()

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
