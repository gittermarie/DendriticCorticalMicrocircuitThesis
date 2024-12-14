# For files with colons
git mv "old:filename.txt" "old-filename.txt"

# Mass rename (in Git Bash or terminal)
find . -type f -name "*:*" -print0 | while IFS= read -r -d '' file; do
    git mv "$file" "$(echo "$file" | sed 's/:/\-/g')"
done