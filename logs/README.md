## Logs

This directory will hold your logs. Stdout (for each epoch) should be routed to its corresponding log file.
Log filenames are in the format:

```training_log_{current_time_utc}_{git_short_commit_id}-{git_branch_name}_E{current_epoch:04}.log```