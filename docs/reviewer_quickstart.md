# SWORD QA Reviewer - Quick Start Guide

## What is SWORD?

SWORD (SWOT River Database) is a global database of rivers used by NASA's SWOT satellite mission. It contains information about ~250,000 river segments ("reaches") worldwide, including their width, slope, flow direction, and classification (river vs. lake vs. canal).

**Your job:** Look at satellite images and verify that the database labels are correct. You'll click buttons to confirm or fix labels. No coding required.

---

## What You'll Be Doing

You'll use a web app (runs in your browser) to review river data. For each issue, you will:

1. Look at a satellite image of the river/lake
2. Read the problem description
3. Click a button: **Fix it**, **Skip**, or classify it

The app saves your work automatically. You can close it and come back later.

---

## Setup (One Time)

### Step 1: Install Python

If you don't have Python installed:

1. Go to <https://www.python.org/downloads/>
2. Download Python 3.10 or newer
3. **Important (Windows):** Check the box that says "Add Python to PATH" during install
4. Click Install

To verify, open Terminal (Mac) or Command Prompt (Windows) and type:

```
python --version
```

You should see something like `Python 3.10.x` or `Python 3.11.x`.

### Step 2: Open Terminal

- **Mac:** Open the "Terminal" app (search for it in Spotlight with Cmd+Space)
- **Windows:** Open "Command Prompt" (search for it in Start menu)

### Step 3: Navigate to the Project Folder

In your terminal, type (replace the path with wherever the folder is on your computer):

```
cd /path/to/SWORD
```

For example, if the folder is on your Desktop:

```
cd ~/Desktop/SWORD
```

### Step 4: Install Dependencies

Type this command (pip is Python's package installer - it downloads the tools we need):

```
pip install -r requirements-reviewer.txt
```

This will download ~6 packages. It may take a minute.

### Step 5: Copy the Database

You'll receive the database file on a USB drive. Copy the file `sword_v17c.duckdb` into:

```
SWORD/data/duckdb/sword_v17c.duckdb
```

The file is about 11 GB, so copying will take a few minutes.

### Step 6: Verify Setup

Run the setup checker to make sure everything is ready:

```
python check_reviewer_setup.py
```

You should see all **PASS** results. If any fail, follow the instructions shown.

### Step 7: Launch the Reviewer

**Option A (recommended):** Double-click `run_reviewer.sh` (Mac) or `run_reviewer.bat` (Windows).

**Option B:** In terminal, type:

```
streamlit run topology_reviewer.py
```

Your browser will open automatically to the reviewer app.

---

## Key Concepts

| Term | What It Means |
|------|---------------|
| **Reach** | One segment of river, typically about 10 km long |
| **Lakeflag** | A label: 0 = river, 1 = lake, 2 = canal, 3 = tidal. Sometimes wrong. |
| **Type** | Another label for the same thing (1 = river, 2 = lake, etc.). Should match lakeflag. |
| **end_reach** | Label for where a river starts/ends: 1 = headwater (start), 2 = outlet (end) |
| **Slope** | How steep the river is. Negative slope = impossible = data error. |
| **facc** | Flow accumulation - how much land drains into this point (km²). Bigger number = bigger river. |
| **Orphan** | A reach with no connections to other reaches. Usually a small pond. |

---

## Tab-by-Tab Guide

The reviewer has several tabs. In **Beginner mode** (on by default), you'll see the most important tabs first.

### Lakeflag/Type Tab

**What you see:** Reaches where the lakeflag and type labels disagree.

**Example:** lakeflag says "lake" but type says "river" - which is correct?

**What to do:**
- Look at the satellite image
- If it looks like a lake (wide, still water), click the button to fix the type to "lake"
- If it looks like a river (narrow, flowing), click **Skip (correct as-is)**

### End Reach Tab

**What you see:** Reaches labeled as headwaters or outlets, but their connections say otherwise.

**Example:** "This reach is labeled as a headwater (where a river starts) but it has upstream neighbors" - that's a labeling error.

**What to do:**
- The app suggests the correct label based on the actual connections
- Click **Fix** to apply the suggested correction
- Click **Skip** if you think the current label is actually correct

### Orphans Tab

**What you see:** Reaches with no connections to any other reaches.

**What to do:**
- Look at the satellite image
- If it's a small isolated pond or lake: click **Valid Orphan**
- If it looks like part of a river that should be connected: click **Needs Connection**

### Slope Tab

**What you see:** Reaches with impossible slope values (negative or extremely steep).

**What to do:**
- **DEM Artifact:** The slope error is from bad elevation data (most common). Click this if it looks like normal river.
- **Looks wrong:** Something seems genuinely wrong with this reach.
- **Skip:** Not sure.

### Suspect Tab

**What you see:** Reaches where automated checks couldn't determine the correct flow accumulation value.

**What to do:** Quick triage with three buttons:
- **FACC looks correct** - the number seems right for the river size
- **FACC is wrong** - the number seems way off
- **Needs more investigation** - not sure

### Fix History Tab

**What you see:** A log of all reviews and fixes made. You can export this as a CSV file.

---

## Decision Rules

**When in doubt, Skip.** It's better to skip 10 ambiguous cases than to miscategorize 1.

- If the satellite image is cloudy or unclear: **Skip**
- If you're not sure whether it's a lake or river: **Skip**
- If the issue doesn't make sense to you: **Skip**
- If it's obviously wrong: **Fix it**

Your skips are still valuable - they tell us which cases are ambiguous and need expert review.

---

## Tips

- **Progress saves automatically.** You can close the browser and come back later.
- **Use the sidebar** to see how many reviews you've completed.
- **Switch regions** using the dropdown in the sidebar (NA = North America, EU = Europe, etc.).
- **The map** shows satellite imagery. Yellow = the reach in question. Orange = upstream. Blue = downstream.
- **Start with NA** (North America) - most familiar geography.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements-reviewer.txt` |
| `FileNotFoundError: sword_v17c.duckdb` | Copy the database to `data/duckdb/` folder |
| Browser doesn't open | Go to `http://localhost:8501` manually |
| App is slow to load | Normal on first load (~30 seconds). Subsequent loads are faster. |
| Map doesn't show satellite images | Check internet connection. Satellite tiles require internet. |
| `python: command not found` | Try `python3` instead of `python` |
| `pip: command not found` | Try `pip3` instead of `pip` |
