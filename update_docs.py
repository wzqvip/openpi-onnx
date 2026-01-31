import os
import datetime

root_dir = "."
date_str = datetime.datetime.now().strftime("%Y-%m-%d")
header = f"<!-- Last Updated: {date_str} -->\n"

for dirpath, dirnames, filenames in os.walk(root_dir):
    # Exclude hidden dirs
    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    if "node_modules" in dirnames:
        dirnames.remove("node_modules")
        
    for filename in filenames:
        if filename.endswith(".md"):
            filepath = os.path.join(dirpath, filename)
            
            # Skip artifacts in .gemini if possible, but user asked for "docs".
            # If path contains .gemini, skip
            if ".gemini" in filepath:
                continue
                
            with open(filepath, "r") as f:
                content = f.read()
                
            if content.startswith("<!-- Last Updated:"):
                # Replace first line
                lines = content.splitlines(keepends=True)
                lines[0] = header
                new_content = "".join(lines)
            else:
                new_content = header + content
                
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"Updated {filepath}")
