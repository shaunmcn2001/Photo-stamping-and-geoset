# Portable Python (optional)

If you want the app to run without relying on Microsoft Store Python, you can place a portable Python in:

`~/Downloads/Photo Processing/Automation Scripts/python/`

The run scripts will use it first if it contains `python.exe` or `pythonw.exe`.

## Recommended approach

Microsoft Store Python is still the simplest and most admin-friendly option. Use portable Python only if you cannot rely on Store Python.

## Build a portable Python folder (Windows)

1. Download the **Windows embeddable zip** for your Python version from https://www.python.org/downloads/windows/
2. Unzip it into a folder named `python` (for example: `C:\Temp\python`).
3. Enable site-packages:
   - Open the `pythonXY._pth` file (for example `python311._pth`).
   - Uncomment `import site`.
   - Add this line at the end:
     - `Lib\site-packages`
4. Download `get-pip.py` from https://bootstrap.pypa.io/get-pip.py
5. Install pip:

```
python.exe get-pip.py
```

6. Install requirements (optional, the app can also install them later):

```
python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt
```

7. Copy the entire `python/` folder into:

`~/Downloads/Photo Processing/Automation Scripts/python/`

## Notes

- The embeddable zip does not include pip by default. The app will warn you if pip is missing.
- If your environment blocks downloads, build the portable Python on another machine and copy the folder over.
