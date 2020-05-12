python setup.py sdist
twine upload dist/* -u echo %TWINEUSER% -p %TWINEPASSWORD%