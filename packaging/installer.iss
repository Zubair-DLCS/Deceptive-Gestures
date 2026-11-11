; Inno Setup script: builds DeceptiveGestures-Setup.exe from the PyInstaller output in dist\DeceptiveGestures
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{9C2F4E61-3B7A-4D58-A1E2-6F0B8C3D5A47}
AppName=Deceptive Gestures
AppVersion={#AppVersion}
AppPublisher=Malak Zubair, G.C.M.S Timergara
AppPublisherURL=https://github.com/Zubair-DLCS/Deceptive-Gestures
DefaultDirName={localappdata}\Programs\DeceptiveGestures
DefaultGroupName=Deceptive Gestures
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=DeceptiveGestures-Setup-{#AppVersion}
SetupIconFile=deceptive_gestures.ico
UninstallDisplayIcon={app}\DeceptiveGestures.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\DeceptiveGestures\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Deceptive Gestures"; Filename: "{app}\DeceptiveGestures.exe"
Name: "{group}\Uninstall Deceptive Gestures"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Deceptive Gestures"; Filename: "{app}\DeceptiveGestures.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\DeceptiveGestures.exe"; Description: "Start Deceptive Gestures now"; Flags: nowait postinstall skipifsilent
