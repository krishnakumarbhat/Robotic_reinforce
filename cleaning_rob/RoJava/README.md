# CleanBot Workspace

## Clone This Repository

To clone this repository **with the submodule initialized**:

```bash
git clone --recurse-submodules git@github.com:LeevAI-Devs/CleanBot.git
```

If you already cloned without submodules:

```bash
cd CleanBot
git submodule update --init --recursive
```

## Fork and Work With Your Own Copy

1. Fork the repository on GitHub.
2. Clone your fork:

```bash
git clone --recurse-submodules git@github.com:<your-username>/CleanBot.git
```

3. If you forgot `--recurse-submodules`:

```bash
cd CleanBot
git submodule update --init --recursive
```

## Make and Push Changes

### Inside the submodule:

```bash
cd src/clean_bot
# Make changes
git add .
git commit -m "Your commit message"
git push origin main  # Or your branch
```

### Then update the parent repo:

```bash
cd ../..
git add src/clean_bot
git commit -m "Update submodule reference"
git push
```

---

Always use `--recurse-submodules` when cloning or working with forks of this repository.
