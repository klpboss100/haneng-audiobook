Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c cd /d """ & scriptDir & """ && python -m streamlit run haneng.py --server.port 8503", 0, False
WScript.Sleep 4000
WshShell.Run "http://localhost:8503"
Set WshShell = Nothing
