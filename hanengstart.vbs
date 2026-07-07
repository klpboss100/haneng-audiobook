Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d ""E:\OneDrive\claude\haneng"" && python -m streamlit run haneng.py --server.port 8503", 0, False
WScript.Sleep 4000
WshShell.Run "http://localhost:8503"
Set WshShell = Nothing
