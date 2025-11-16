import psutil
import cpuinfo

def get_cpu_info():
    """
    Retrieves a formatted summary of the system's CPU information.

    Uses the `cpuinfo` and `psutil` libraries to extract:
    - CPU model/brand name
    - Number of logical (hyper-threaded) cores
    - Number of physical cores

    Returns
    -------
    str
        A formatted multi-line string containing the CPU model name,
        number of logical cores, and number of physical cores.

    Notes
    -----
    - Requires the `py-cpuinfo` and `psutil` libraries.
    - On some systems, the CPU model name may be returned as "Unknown CPU"
      if not detected properly.
    """
    # CPU model name
    info = cpuinfo.get_cpu_info()
    cpu_model = info.get('brand_raw', 'Unknown CPU')
    # Number of logical and physical cores
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    txt = (
         "\n"
        f"{'CPU Model:':25} {cpu_model}\n"
        f"{'Logical cores (threads):':25} {logical_cores}\n"
        f"{'Physical cores:':25} {physical_cores}\n"
    )
    return txt
