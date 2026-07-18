import sys
import subprocess
result = subprocess.run([sys.executable, "-m", "torchfits.cli", "probe", "http://127.0.0.1/test.fits"], capture_output=True, text=True)
print("RETURN CODE", result.returncode)
print("STDOUT", result.stdout)
print("STDERR", result.stderr)
