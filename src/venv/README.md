Step 1: Package Installation
** Windows Installation **
Run setup.bat
```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```


** MacOS/Linux Installation **
Run setup.sh
```python
#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Input the labelled images to be processed in the `Raw_Images` folder. 
Review the selected methods of operation in `SpecificationParametersModule.py`. Once satisfied,  run the `ControlModule.py` file.
