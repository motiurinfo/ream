#!/usr/bin/env python3
"""
hardware_info.py  --  collect hardware + software details for the paper's
"Computational cost" section.

Run this ON THE MACHINE THAT RAN THE EXPERIMENTS:

    python hardware_info.py

It prints a human-readable report and also writes hardware_info.json next to
itself. Standard library only; the extras (CPU model name, exact RAM, library
versions) are filled in when those packages/commands are available and are
skipped cleanly when they are not.
"""

import json, platform, os, re, subprocess, sys
from datetime import datetime


def _run(cmd):
    """Run a shell command, return stripped stdout or None on any failure."""
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        return out.decode(errors="replace").strip()
    except Exception:
        return None


def cpu_model():
    """Human-readable CPU model name, across Linux / macOS / Windows."""
    sysname = platform.system()
    if sysname == "Linux":
        txt = _run("cat /proc/cpuinfo")
        if txt:
            m = re.search(r"model name\s*:\s*(.+)", txt)
            if m:
                return m.group(1).strip()
        return _run("lscpu | grep -i 'model name'")
    if sysname == "Darwin":
        return _run("sysctl -n machdep.cpu.brand_string")
    if sysname == "Windows":
        return _run("wmic cpu get name") or platform.processor()
    return platform.processor() or None


def core_counts():
    """(physical_cores, logical_cores). Physical needs psutil; falls back to None."""
    logical = os.cpu_count()
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        sysname = platform.system()
        if sysname == "Linux":
            # count unique physical ids x cores per socket
            v = _run("lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l")
            physical = int(v) if v and v.isdigit() else None
        elif sysname == "Darwin":
            v = _run("sysctl -n hw.physicalcpu")
            physical = int(v) if v and v.isdigit() else None
    return physical, logical


def cpu_freq_ghz():
    """Max/base clock in GHz if obtainable, else None."""
    try:
        import psutil
        f = psutil.cpu_freq()
        if f and f.max:
            return round(f.max / 1000.0, 2)
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Linux":
        v = _run("lscpu | grep -i 'max mhz'")
        if v:
            m = re.search(r"([\d.]+)", v)
            if m:
                return round(float(m.group(1)) / 1000.0, 2)
    if sysname == "Darwin":
        v = _run("sysctl -n hw.cpufrequency_max")  # Hz, Intel Macs only
        if v and v.isdigit():
            return round(int(v) / 1e9, 2)
    return None


def ram_gb():
    """Total physical RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Linux":
        v = _run("grep MemTotal /proc/meminfo")  # in kB
        if v:
            m = re.search(r"(\d+)", v)
            if m:
                return round(int(m.group(1)) / (1024 ** 2), 1)
    if sysname == "Darwin":
        v = _run("sysctl -n hw.memsize")  # bytes
        if v and v.isdigit():
            return round(int(v) / (1024 ** 3), 1)
    return None


def gpu_info():
    """List NVIDIA GPUs if nvidia-smi is present; else None (CPU-only run)."""
    v = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if v:
        return [line.strip() for line in v.splitlines() if line.strip()]
    return None


def lib_versions():
    """Versions of the libraries that drive the cost numbers."""
    libs = ["shap", "lightgbm", "sklearn", "lime", "numpy", "scipy", "pandas"]
    out = {}
    for name in libs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None  # not installed
    return out


def main():
    phys, logical = core_counts()
    info = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "cpu_model": cpu_model(),
        "physical_cores": phys,
        "logical_cores": logical,
        "cpu_max_ghz": cpu_freq_ghz(),
        "ram_gb": ram_gb(),
        "gpu": gpu_info(),
        "os": f"{platform.system()} {platform.release()}",
        "os_detail": platform.platform(),
        "python": platform.python_version(),
        "libraries": lib_versions(),
    }

    # ---- human-readable report ----
    print("=" * 60)
    print("HARDWARE / SOFTWARE REPORT  (for the paper's cost section)")
    print("=" * 60)
    print(f"CPU            : {info['cpu_model']}")
    print(f"Cores          : {info['physical_cores']} physical / "
          f"{info['logical_cores']} logical")
    print(f"Max clock      : "
          f"{str(info['cpu_max_ghz']) + ' GHz' if info['cpu_max_ghz'] else 'unknown'}")
    print(f"RAM            : "
          f"{str(info['ram_gb']) + ' GB' if info['ram_gb'] else 'unknown'}")
    print(f"GPU            : {info['gpu'] if info['gpu'] else 'none / CPU-only'}")
    print(f"OS             : {info['os']}")
    print(f"Python         : {info['python']}")
    print("Libraries      :")
    for k, v in info["libraries"].items():
        print(f"   {k:10s}: {v if v else '(not installed)'}")
    print("=" * 60)

    # ---- ready-to-paste sentence for the paper ----
    cores = (f"{info['physical_cores']}-core "
             if info['physical_cores'] else "")
    ghz = (f" at {info['cpu_max_ghz']} GHz"
           if info['cpu_max_ghz'] else "")
    ram = (f", {info['ram_gb']} GB RAM" if info['ram_gb'] else "")
    cpu = info["cpu_model"] or "an unspecified CPU"
    shap_v = info["libraries"].get("shap")
    shap_s = f", SHAP {shap_v}" if shap_v else ""
    print("\nPaste-ready sentence:")
    print(f'  "All timings were measured on a {cores}{cpu}{ghz}{ram}, '
          f'running Python {info["python"]}{shap_s}."')

    with open("hardware_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nSaved full detail to hardware_info.json")


if __name__ == "__main__":
    main()
