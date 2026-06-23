import os
import shutil


def clone_avd(
    src_avd_name,
    tar_avd_name,
    src_android_avd_home,
    tar_android_avd_home,
    src_sdk,
    tar_sdk,
    src_avd_dir=None,
    src_ini_file=None,
    tar_avd_dir=None,
    tar_ini_file=None,
    target_linux=False,
):
    """Clone the source AVD to the target AVD.

    Parameters:
    - src_avd_name: The name of the source AVD folder as stored in the source AVD.
    - tar_avd_name: The name of the target AVD folder to be updated to.
    - src_android_avd_home: The path that contains the source AVD folder as stored in the source AVD.
    - tar_android_avd_home: The path that contains the target AVD folder to be updated to.
    - src_sdk: The Android SDK path stored in the source AVD.
    - tar_sdk: The Android SDK path to be updated to.
    - src_avd_dir: The actual path of the AVD folder stored on this machine.
    - src_ini_file: The actual path of the .ini file stored on this machine.
    - tar_avd_dir: The actual path of the AVD folder stored on this machine.
    - tar_ini_file: The actual path of the .ini file stored on this machine.

    This function copies the source AVD folder and its .ini file to a new target AVD
    and updates the paths inside the .ini files accordingly.
    """

    # Paths for source and target AVD directories and .ini files
    src_avd_dir = src_avd_dir if src_avd_dir else os.path.join(src_android_avd_home, src_avd_name + ".avd")
    src_ini_file = src_ini_file if src_ini_file else os.path.join(src_android_avd_home, src_avd_name + ".ini")
    tar_avd_dir = tar_avd_dir if tar_avd_dir else os.path.join(tar_android_avd_home, tar_avd_name + ".avd")
    tar_ini_file = tar_ini_file if tar_ini_file else os.path.join(tar_android_avd_home, tar_avd_name + ".ini")

    # Copy the AVD folder
    print(f"Copying the AVD folder from {src_avd_dir} to {tar_avd_dir}")
    if os.path.exists(tar_avd_dir):
        print("Target AVD exists")
        return
    else:
        shutil.copytree(src_avd_dir, tar_avd_dir)

    # Copy the .ini file and modify it for the new AVD
    with open(src_ini_file) as src_ini, open(tar_ini_file, "w") as tar_ini:
        for line in src_ini:
            new_line = line.replace(src_avd_name, tar_avd_name).replace(src_android_avd_home, tar_android_avd_home)
            if target_linux:
                new_line = new_line.replace("\\", "/")
            tar_ini.write(new_line)

    # Update paths inside the target AVD's .ini files
    for ini_name in ["config.ini", "hardware-qemu.ini"]:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        if os.path.exists(ini_path):
            with open(ini_path) as file:
                lines = file.readlines()
            with open(ini_path, "w") as file:
                for line in lines:
                    # Update paths and AVD name/ID
                    new_line = (
                        line.replace(src_avd_name, tar_avd_name)
                        .replace(src_android_avd_home, tar_android_avd_home)
                        .replace(src_sdk, tar_sdk)
                    )
                    if target_linux:
                        new_line = new_line.replace("\\", "/")
                    file.write(new_line)

    # Update the snapshots' hardware.ini file if it exists
    for snapshot_name in os.listdir(os.path.join(tar_avd_dir, "snapshots")):
        snapshots_hw_ini = os.path.join(tar_avd_dir, "snapshots", snapshot_name, "hardware.ini")
        if os.path.exists(snapshots_hw_ini):
            with open(snapshots_hw_ini) as file:
                lines = file.readlines()
            with open(snapshots_hw_ini, "w") as file:
                for line in lines:
                    # Update AVD name/ID
                    new_line = (
                        line.replace(src_avd_name, tar_avd_name)
                        .replace(src_android_avd_home, tar_android_avd_home)
                        .replace(src_sdk, tar_sdk)
                    )
                    if target_linux:
                        new_line = new_line.replace("\\", "/")
                    file.write(new_line)


if __name__ == "__main__":
    clone_avd(
        src_avd_dir=os.path.join(os.getcwd(), r".\avd\Benchmark-Eng-API32-12GB.avd"),
        src_ini_file=os.path.join(os.getcwd(), r".\avd\Benchmark-Eng-API32-12GB.ini"),
        src_avd_name=r"Benchmark-Eng-API32-12GB",
        tar_avd_name=r"Benchmark-Eng-API32-12GB",
        src_android_avd_home=r"C:\Users\User\.android\avd",
        tar_android_avd_home=r"C:\Users\User\.android\avd",  # Update this according to your machine env
        src_sdk=r"C:\Users\User\AppData\Local\Android\Sdk",
        tar_sdk=r"C:\Users\User\AppData\Local\Android\Sdk",  # Update this according to your machine env
        target_linux=os.name == "posix",
    )

    # After executing this function, you should be able to launch the emulator via
    # `emulator -avd Benchmark-Eng-API32-12GB`
