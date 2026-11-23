#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\release\Scenespy-windows-x64"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\release"
#endif

#define AppName "Scenespy"
#define AppPublisher "Scenespy"
#define AppExeName "Scenespy.exe"

[Setup]
AppId={{F61E81CA-72A1-44B6-AE70-A819AC886091}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=Scenespy-{#AppVersion}-windows-x64-setup
SetupIconFile=..\..\scenespy\assets\images\exe-icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ExtraDiskSpaceRequired=629145600
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "aifeatures"; Description: "Install AI face detection features (recommended; up to 6 GB total disk space)"; GroupDescription: "Optional features:"; Flags: checkedonce

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_runtime_windows.ps1"" -TorchMode cpu -SkipAI"; StatusMsg: "Downloading and installing required Scenespy components..."; Flags: runhidden waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_runtime_windows.ps1"" -TorchMode auto -SkipFFmpeg"; StatusMsg: "Downloading optional AI face detection features..."; Flags: runhidden waituntilterminated; Tasks: aifeatures
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure InitializeWizard();
begin
  WizardForm.WelcomeLabel2.Caption := WizardForm.WelcomeLabel2.Caption + #13#10 + #13#10 +
    'Disk space: core installation requires approximately 1 GB. ' +
    'Optional AI features require up to 3 GB on CPU systems or up to 6 GB with NVIDIA/CUDA.';
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  RuntimeSpace: String;
begin
  if WizardIsTaskSelected('aifeatures') then
    RuntimeSpace := 'Up to 6 GB total (AI features selected; hardware-specific runtime is chosen automatically)'
  else
    RuntimeSpace := 'Approximately 1 GB total (AI features not selected)';
  Result := MemoUserInfoInfo + MemoDirInfo + MemoTypeInfo +
    MemoComponentsInfo + MemoGroupInfo + MemoTasksInfo + NewLine + NewLine +
    'Estimated disk space after setup:' +
    NewLine + RuntimeSpace;
end;
