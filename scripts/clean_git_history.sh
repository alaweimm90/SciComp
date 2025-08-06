#!/bin/bash
#
# Script to remove Claude attribution from Git history
# USE WITH CAUTION: This will rewrite Git history
#

set -e

echo "🧹 Berkeley SciComp Framework - Git History Cleaner"
echo "=================================================="
echo
echo "This script will remove Claude attribution from Git history."
echo "WARNING: This will rewrite commit history and force-push is required."
echo

read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

echo "🔍 Backing up current repository..."
cd "$(dirname "$0")/.."
git bundle create backup-before-clean.bundle --all

echo "🧹 Removing Claude attribution from commit messages..."

# Use git filter-branch to rewrite commit messages
git filter-branch -f --msg-filter '
sed \
-e "/Co-authored-by: Claude/d" \
-e "/Co-Authored-By: Claude/d" \
-e "/Generated with.*Claude/d" \
-e "/claude\.ai/d" \
-e "/🤖 Generated with \[Claude Code\]/d" \
-e "/🤖.*Claude/d"
' HEAD

echo "🔄 Cleaning up filter-branch references..."
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

echo "🗑️  Expiring reflog and garbage collecting..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ Git history cleaned successfully!"
echo 
echo "📋 Next steps:"
echo "1. Verify the commit history looks correct: git log --oneline -10"  
echo "2. If satisfied, force-push to remote: git push --force-with-lease origin main"
echo "3. Notify collaborators that history has been rewritten"
echo
echo "⚠️  Backup created: backup-before-clean.bundle"
echo "   To restore if needed: git clone backup-before-clean.bundle repo-restored"
echo

echo "🐻 Berkeley SciComp Framework - History cleaned! Go Bears!"