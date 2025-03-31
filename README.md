# Zoom Attendance Report Tool

This tool generates a visual and CSV-based attendance report from Zoom participant exports. It filters for specific days/times and organizes names for consistency and readability.

---

## 📦 What It Does

- Filters Zoom meetings by:
  - Sundays (all day)
  - Tuesday and Thursday evenings (6:00 PM – 11:00 PM)
- Normalizes and consolidates participant names
- Optionally clusters similar names using AI (if supported)
- Displays results in a clean, read-only table with alternating row colors
- Outputs a CSV report showing attendance symbols:
  - `●` for weekday attendance (Tue/Thu evenings)
  - `■` for weekend attendance (Sundays)

---

## 📤 How to Export Attendance from Zoom

1. Log in to your Zoom account.
2. Expand **Account Management** in the left sidebar, under the _ADMIN_ section.
3. Click on **Reports**.
4. On the right side, click the **Usage reports** tab.
5. Click the **Meeting and webinar history** report.
6. Set the date range.
7. Click **Search**.
8. Click the **Export** button.
9. Choose the **Export the list with participants details** option from the menu.

Save this file and use it as input to the script.

---

## 🧰 Requirements

You can install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
