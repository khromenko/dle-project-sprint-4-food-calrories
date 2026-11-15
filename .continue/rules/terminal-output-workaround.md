# Rule: Forward terminal output to a temp file in remote/devcontainer sessions

## Context

When working inside a VS Code **devcontainer / remote (Linux) environment**, the
Continue `run_terminal_command` tool executes commands in the remote terminal, but
**stdout capture is not returned to the model**. The tool reports:

> Command executed in remote terminal. Output capture is not yet available for remote environments.

This means commands like `dir`, `ls -la`, `pytest`, `git status`, etc. run correctly
but their output is **invisible to the assistant**.

## Decision / Workaround (ALWAYS DO THIS)

In a remote/devcontainer environment, when you need to READ the output of a shell
command, do **not** rely on `run_terminal_command` capturing stdout. Instead,
capture it into a **dedicated temp directory in the project root**:

```
.agents_tmp_terminal_out/      # all captured terminal output goes here
```

1. **Ensure the dir exists and redirect the command output into it**, capturing
   both stdout and stderr, e.g.:

   ```bash
   mkdir -p .agents_tmp_terminal_out
   <command> > .agents_tmp_terminal_out/<name>.txt 2>&1
   ```

   Examples:
   ```bash
   mkdir -p .agents_tmp_terminal_out
   dir > .agents_tmp_terminal_out/dirlist.txt 2>&1
   ls -la > .agents_tmp_terminal_out/ls.txt 2>&1
   git status > .agents_tmp_terminal_out/git-status.txt 2>&1
   python -m pytest > .agents_tmp_terminal_out/pytest.txt 2>&1
   ```

2. **Read the file back** with the `read_file` tool (which works fine in remote
   environments), e.g. `read_file(".agents_tmp_terminal_out/dirlist.txt")`.

3. **Clean up** when done (optional but tidy):
   ```bash
   rm -rf .agents_tmp_terminal_out
   ```

## Naming convention

Give each captured command a short, descriptive filename inside the temp dir:

| Command            | Output file                                   |
|--------------------|-----------------------------------------------|
| `dir` / `ls`       | `.agents_tmp_terminal_out/dirlist.txt`        |
| `git status`       | `.agents_tmp_terminal_out/git-status.txt`     |
| `pytest`           | `.agents_tmp_terminal_out/pytest.txt`         |
| `pip install ...`  | `.agents_tmp_terminal_out/pip-install.txt`    |

## When to use

- Anytime you are in a devcontainer / remote terminal AND you need to *see* the
  command's output to make a decision.
- If the command is purely for side effects (install, build) and its output isn't
  needed, a direct `run_terminal_command` is fine.

## Notes / troubleshooting

- Prefer Continue's native file tools (`ls`, `read_file`, `file_glob_search`) for
  directory listings and reading files — they work reliably without redirection.
- Running Continue "inside" the devcontainer does NOT fix this; it already runs there.
  The limitation is the remote-terminal stdout bridge, not the environment location.
- `.devcontainer/devcontainer.json` in this project is clean (no custom terminal
  profile, no root remoteUser) — it is NOT the cause of the missing capture.
- Add `.agents_tmp_terminal_out/` to `.gitignore` so captured output isn't committed.
- Consider updating VS Code + the Continue extension; remote output capture support
  has improved in newer versions.
