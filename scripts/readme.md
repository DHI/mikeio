# Scripts

[`uv run`](https://docs.astral.sh/uv/reference/cli/#uv-run) allows you to run Python scripts without any hassle.

Below is an example of using MIKE IO to concatenate files:

```bash
$ uv run https://raw.githubusercontent.com/DHI/mikeio/refs/heads/main/scripts/concat.py tide* out.dfs1
Installed 20 packages in 59ms
Concatenating files: ['tide1.dfs1', 'tide12.dfs1', 'tide2.dfs1', 'tide2_offset.dfs1', 'tide4.dfs1']
100%|███████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 1035.63it/s]
Created: out.dfs1
```

```bash
$ uv run https://raw.githubusercontent.com/DHI/mikeio/refs/heads/main/scripts/concat.py tide1.dfs1 tide2.dfs1 out2.dfs1
Concatenating files: ['tide1.dfs1', 'tide2.dfs1']
100%|███████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 1301.97it/s]
Created: out2.dfs1
```
