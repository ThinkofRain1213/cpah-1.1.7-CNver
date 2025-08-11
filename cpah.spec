# -*- mode: python ; coding: utf-8 -*-

import ahk, inspect, pathlib  ## oh god, what is he doing

ahk_dir = pathlib.Path(inspect.getfile(ahk)).parent.resolve()  ## oh
ahk_script_files = (
    "keyboard/send_input.ahk",
    "mouse/mouse_position.ahk",
    "base.ahk",
)

add_data = [
    ("cpah/resources", "resources/."),
    ("VERSION", "."),
]

for script_file in ahk_script_files:  ## oh NO
    add_data.append(
        (
            str(ahk_dir / "templates" / script_file),
            str((pathlib.Path("ahk/templates") / script_file).parent),
        )
    )

block_cipher = None


a = Analysis(['entrypoint.py'],
             binaries=[],
             datas=add_data,
             hiddenimports=['PySide2.QtXml'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='cpah',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='cpah\\resources\\images\\icon.ico')
