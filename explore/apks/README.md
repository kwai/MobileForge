# Local APK Cache

This directory stores pre-downloaded APK files used during device initialization. Keeping APKs locally avoids repeated downloads and makes AndroidWorld-style setup faster and more reproducible.

## Layout

```text
apks/
|-- README.md
|-- apk_manifest.json
|-- com.arduia.expense.apk
|-- net.gsantner.markor.apk
`-- ...
```

## Getting APK Files

Option 1: export APKs from a connected device.

```bash
python scripts/export_apks.py
python scripts/export_apks.py -s emulator-5554
python scripts/export_apks.py -p com.arduia.expense
```

Option 2: download APKs manually from sources such as APKPure, APKMirror, or F-Droid, then place them in this directory.

Use package-name filenames:

```text
com.arduia.expense.apk
net.gsantner.markor.apk
```

Update `apk_manifest.json` after adding files:

```json
{
  "apps": {
    "com.arduia.expense": {
      "filename": "com.arduia.expense.apk",
      "version": "1.0.0",
      "source": "exported_from_device"
    }
  }
}
```

## App Coverage

Android system apps are usually preinstalled, including Camera, Settings, Clock, Contacts, Dialer, and DocumentsUI. Third-party APKs commonly used by AndroidWorld and MobileForge include Expense, Markor, OsmAnd, Simple Calendar, Simple SMS Messenger, Simple Gallery, Broccoli, Tasks, Joplin, OpenTracks, Retro Music, and VLC.

## Installation Priority

During device initialization, MobileForge first checks this local cache. If an APK is missing, the setup falls back to the AndroidWorld download path.

| Setup mode | Typical time |
| --- | --- |
| Network download for all APKs | 10-15 minutes |
| Local APK cache | 2-3 minutes |

## Notes

- Use APK versions compatible with the AndroidWorld benchmark tasks.
- Keep signatures consistent across versions of the same package.
- A complete APK cache usually requires about 500 MB to 1 GB of disk space.
